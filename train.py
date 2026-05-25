#!/usr/bin/env python
"""StyleFlow LoRA training (channel-wise seed concatenation).

This script trains a LoRA adapter on top of FLUX.1-dev so that the
MM-DiT consumes the seed garment as additional input channels, as
described in the paper. The recipe follows the published configuration
(joint training on the three CIG benchmarks),
rank-16 LoRA on attention + FFN + the widened x_embedder, cosine LR,
20-step Euler inference at evaluation time).

The custom ``save_model_hook`` follows Diffusers' reference pattern: it
iterates over the models list *without* mutating it and pops the
matching ``weights`` entry. Every periodic checkpoint therefore
contains a valid ``pytorch_lora_weights.safetensors`` and can be
resumed via ``--resume_from_checkpoint``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils.torch_utils import is_compiled_module

from styleflow.data import FashionDataset, collate_fn
from styleflow.pipeline import StyleFlowPipeline, expand_x_embedder_for_seed_concat


logger = get_logger(__name__)


def parse_args(input_args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--logging_dir", default="logs")

    # Data
    p.add_argument("--datasets_root", required=True)
    p.add_argument("--datasets", default="fashionvc,expreduced,fashiontaobaotb")
    p.add_argument("--train_csv_name", default="train_full_columns_dif_G.csv")
    p.add_argument("--prompt_columns", default="bottom_description")
    p.add_argument("--max_examples_per_dataset", type=int, default=None)
    p.add_argument("--subset_seed", type=int, default=0)

    # Optim / schedule
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_train_steps", type=int, default=93483)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--scale_lr", action="store_true")
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--lr_num_cycles", type=int, default=1)
    p.add_argument("--lr_power", type=float, default=1.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # Precision
    p.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--upcast_before_saving", action="store_true")

    # LoRA
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_layers", default=None,
                   help="Comma-separated, 'all-linear', or unset for default attention+FFN+x_embedder.")
    p.add_argument("--gaussian_init_lora", action="store_true")
    p.add_argument("--use_lora_bias", action="store_true")

    # Flow matching
    p.add_argument("--weighting_scheme", default="none")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--proportion_empty_prompts", type=float, default=0.0)

    # Checkpointing
    p.add_argument("--validation_steps", type=int, default=1000)
    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--checkpoints_total_limit", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    # Tracking
    p.add_argument("--report_to", default="tensorboard")
    p.add_argument("--tracker_project_name", default="styleflow")

    return p.parse_args(input_args)


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ----- Models -----
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)
    vae.to(dtype=torch.float32)  # keep VAE in fp32 for stable encoding
    flux_transformer.to(dtype=weight_dtype, device=accelerator.device)

    # Widen x_embedder so it accepts [noise || seed]
    expand_x_embedder_for_seed_concat(flux_transformer)

    # ----- LoRA -----
    if args.lora_layers is None:
        target_modules = [
            "x_embedder",
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
        ]
    elif args.lora_layers == "all-linear":
        target_modules = sorted({n for n, m in flux_transformer.named_modules() if isinstance(m, torch.nn.Linear)})
    else:
        target_modules = [s.strip() for s in args.lora_layers.split(",")]
        if "x_embedder" not in target_modules:
            target_modules.append("x_embedder")

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        lora_bias=args.use_lora_bias,
    )
    flux_transformer.add_adapter(lora_cfg)

    def unwrap_model(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    transformer_cls = type(unwrap_model(flux_transformer))

    # ----- Save / load hooks (no `models.remove` mutation) -----
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if not accelerator.is_main_process:
                return
            transformer_lora_layers = None
            for model in models:
                m = unwrap_model(model)
                if isinstance(m, transformer_cls):
                    transformer_lora_layers = get_peft_model_state_dict(m)
                if weights:
                    weights.pop()
            if transformer_lora_layers is not None:
                StyleFlowPipeline.save_lora_weights(
                    output_dir, transformer_lora_layers=transformer_lora_layers
                )

        def load_model_hook(models, input_dir):
            for model in list(models):
                m = unwrap_model(model)
                if isinstance(m, transformer_cls):
                    lora_path_st = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
                    lora_path_bin = os.path.join(input_dir, "pytorch_lora_weights.bin")
                    if os.path.exists(lora_path_st) or os.path.exists(lora_path_bin):
                        lora_state = StyleFlowPipeline.lora_state_dict(input_dir)
                        transformer_state = {
                            k.replace("transformer.", ""): v
                            for k, v in lora_state.items()
                            if k.startswith("transformer.") and "lora" in k
                        }
                        set_peft_model_state_dict(m, transformer_state, adapter_name="default")
                    else:
                        accelerator.print(f"[load] no LoRA weights in {input_dir}; transformer unchanged.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.mixed_precision == "fp16":
        cast_training_params([flux_transformer], dtype=torch.float32)
    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
    else:
        opt_cls = torch.optim.AdamW

    trainable = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    optimizer = opt_cls(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ----- Data -----
    dataset_names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    prompt_columns = [c.strip() for c in args.prompt_columns.split(",") if c.strip()]
    train_dataset = FashionDataset(
        datasets_root=args.datasets_root,
        dataset_names=dataset_names,
        train_csv_name=args.train_csv_name,
        prompt_columns=prompt_columns,
        max_examples_per_dataset=args.max_examples_per_dataset,
        subset_seed=args.subset_seed,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  num examples = {len(train_dataset)}")
    logger.info(f"  num epochs = {args.num_train_epochs}")
    logger.info(f"  per-device bs = {args.train_batch_size}")
    logger.info(f"  total bs (incl. grad-accum) = {total_batch_size}")
    logger.info(f"  optimization steps = {args.max_train_steps}")

    # ----- Text encoding pipeline (CLIP + T5 only) -----
    from styleflow.pipeline import StyleFlowPipeline as _SP
    text_encoding_pipeline = _SP.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype,
    )

    # ----- Resume -----
    global_step, first_epoch, initial_global_step = 0, 0, 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]),
            )
            path = dirs[-1] if dirs else None
        else:
            path = os.path.basename(args.resume_from_checkpoint)
        if path is not None:
            logger.info(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _encode_images(pixels):
        z = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
        return z.to(weight_dtype)

    # ----- Training loop -----
    for epoch in range(first_epoch, args.num_train_epochs):
        flux_transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(flux_transformer):
                pixel_latents = _encode_images(batch["pixel_values"])
                control_latents = _encode_images(batch["conditioning_pixel_values"])

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz, logit_mean=args.logit_mean,
                    logit_std=args.logit_std, mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy = (1.0 - sigmas) * pixel_latents + sigmas * noise

                latent_in = torch.cat([noisy, control_latents], dim=1)
                packed = StyleFlowPipeline._pack_latents(
                    latent_in, batch_size=bsz,
                    num_channels_latents=latent_in.shape[1],
                    height=latent_in.shape[2], width=latent_in.shape[3],
                )
                latent_image_ids = StyleFlowPipeline._prepare_latent_image_ids(
                    bsz, latent_in.shape[2] // 2, latent_in.shape[3] // 2,
                    accelerator.device, weight_dtype,
                )

                # Text encoding (CLIP + T5)
                with torch.no_grad():
                    text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
                    prompt_embeds, pooled, text_ids = text_encoding_pipeline.encode_prompt(
                        batch["captions"], prompt_2=None
                    )
                if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                    prompt_embeds.zero_(); pooled.zero_()

                base = unwrap_model(flux_transformer)
                guidance_vec = (
                    torch.full((bsz,), args.guidance_scale, device=pixel_latents.device, dtype=weight_dtype)
                    if getattr(base.config, "guidance_embeds", False)
                    else None
                )

                model_pred = flux_transformer(
                    hidden_states=packed,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = StyleFlowPipeline._unpack_latents(
                    model_pred,
                    height=noisy.shape[2] * vae_scale_factor,
                    width=noisy.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - pixel_latents
                loss = ((weighting.float() * (model_pred.float() - target.float()) ** 2)
                        .reshape(target.shape[0], -1)).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and (global_step % args.checkpointing_steps == 0):
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        excess = max(0, len(ckpts) - args.checkpoints_total_limit + 1)
                        for d in ckpts[:excess]:
                            shutil.rmtree(os.path.join(args.output_dir, d))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # ----- Final save (outside the hook; always runs on clean exit) -----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        m = unwrap_model(flux_transformer)
        if args.upcast_before_saving:
            m.to(torch.float32)
        lora_layers = get_peft_model_state_dict(m)
        StyleFlowPipeline.save_lora_weights(
            save_directory=args.output_dir, transformer_lora_layers=lora_layers
        )
        del flux_transformer, text_encoding_pipeline, vae
        free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    main()
