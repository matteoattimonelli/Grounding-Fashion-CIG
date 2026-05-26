"""Standalone Diffusers pipeline for StyleFlow.

StyleFlow conditions FLUX's MM-DiT on (i) the textual instruction via
FLUX's native dual text encoders (CLIP + T5) and (ii) the seed garment
image via channel-wise concatenation of its VAE latent with the noise
input. The ``x_embedder`` projection of the MM-DiT is widened from
``in_channels`` to ``2 * in_channels`` to accept the concatenated
representation; the right half is zero-initialised so that, at training
start, the model reproduces the pretrained text-to-image behaviour.

"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


# ---------------------------------------------------------------------------
# Helpers (mirror the FLUX implementation; kept inline so this file is
# fully self-contained and does not import private symbols from
# ``diffusers.pipelines.flux``).
# ---------------------------------------------------------------------------
def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent shift used by FLUX's flow-matching scheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Tiny wrapper around ``scheduler.set_timesteps`` that handles the
    different ``timesteps``/``sigmas``/``num_inference_steps`` argument
    combinations Diffusers schedulers accept."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_ts = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_ts:
            raise ValueError("Scheduler does not support a custom `timesteps` schedule.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sig = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sig:
            raise ValueError("Scheduler does not support a custom `sigmas` schedule.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# Public helper: widen the MM-DiT input projection to accept the
# concatenated [noise | seed-latent] input. Right half is zero-init.
# ---------------------------------------------------------------------------
@torch.no_grad()
def expand_x_embedder_for_seed_concat(transformer: FluxTransformer2DModel) -> FluxTransformer2DModel:
    """Double ``x_embedder.in_features`` (in place) so the transformer
    accepts ``[noise || seed_latent]`` along the channel axis. The
    pretrained left-half weights are preserved exactly; the right half
    is zero-initialised so the model starts equivalent to the pretrained
    text-to-image behaviour."""
    initial = transformer.config.in_channels
    if transformer.x_embedder.in_features == initial * 2:
        return transformer  # already widened
    if transformer.x_embedder.in_features != initial:
        raise ValueError(
            f"Unexpected x_embedder width: {transformer.x_embedder.in_features} "
            f"(expected {initial} or {initial * 2})."
        )
    new_lin = nn.Linear(
        initial * 2,
        transformer.x_embedder.out_features,
        bias=transformer.x_embedder.bias is not None,
        dtype=transformer.dtype,
        device=transformer.device,
    )
    new_lin.weight.zero_()
    new_lin.weight[:, :initial].copy_(transformer.x_embedder.weight)
    if transformer.x_embedder.bias is not None:
        new_lin.bias.copy_(transformer.x_embedder.bias)
    transformer.x_embedder = new_lin
    transformer.register_to_config(in_channels=initial * 2, out_channels=initial)
    return transformer


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class StyleFlowPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """Standalone StyleFlow inference pipeline.

    Composes the standard FLUX components (CLIP/T5 text encoders, VAE,
    MM-DiT transformer, flow-matching scheduler) and runs Rectified
    Flow Matching sampling with channel-wise seed-image conditioning.

    Use :meth:`from_pretrained` to load the FLUX backbone, then call
    :meth:`load_lora_weights` (inherited from ``FluxLoraLoaderMixin``)
    on a directory containing ``pytorch_lora_weights.safetensors``.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.vae_latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        # FLUX packs latents into 2x2 patches; account for that in the image processor.
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.vae_latent_channels,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 77
        )
        self.default_sample_size = 128

    # -----------------------------------------------------------------
    # Patch packing / unpacking (mirrors FLUX). Inlined to avoid
    # depending on private symbols from diffusers.pipelines.flux.
    # -----------------------------------------------------------------
    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    # -----------------------------------------------------------------
    # Text encoding (FLUX dual-encoder: CLIP pooled + T5 sequence).
    # -----------------------------------------------------------------
    def _get_clip_prompt_embeds(self, prompt: Union[str, List[str]], num_images_per_prompt: int, device, dtype):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        bsz = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
        prompt_embeds = prompt_embeds.pooler_output.to(dtype=dtype, device=device)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt).view(bsz * num_images_per_prompt, -1)
        return prompt_embeds

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        max_sequence_length: int,
        device,
        dtype,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        bsz = len(prompt)
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bsz * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        device = device or self._execution_device
        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 if prompt_2 is not None else prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        pooled_prompt_embeds = self._get_clip_prompt_embeds(prompt, num_images_per_prompt, device, dtype)
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt_2, num_images_per_prompt, max_sequence_length, device, dtype
        )
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # -----------------------------------------------------------------
    # Latent and seed-image preparation.
    # -----------------------------------------------------------------
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        return latents, latent_image_ids

    def prepare_seed_image(
        self,
        image: PipelineImageInput,
        width: int,
        height: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Resize/normalise the seed image into a tensor suitable for VAE encoding."""
        if isinstance(image, torch.Tensor):
            img = image
        else:
            img = self.image_processor.preprocess(image, height=height, width=width)
        img = img.repeat_interleave(batch_size // img.shape[0], dim=0) if img.shape[0] < batch_size else img
        return img.to(device=device, dtype=dtype)

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """Run StyleFlow inference.

        ``control_image`` is the seed garment image. It is VAE-encoded
        and channel-concatenated with the noisy bottom latent at every
        denoising step. The MM-DiT must have been widened with
        :func:`expand_x_embedder_for_seed_concat` (this happens
        automatically inside :func:`build_styleflow_pipeline`).
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        device = self._execution_device
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # --- 1. Encode the text ---
        if prompt_embeds is None or pooled_prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)

        # --- 2. Encode the seed image to packed control tokens ---
        if control_image is None:
            raise ValueError(
                "`control_image` is required: StyleFlow conditions on a seed garment image."
            )
        seed_pixels = self.prepare_seed_image(
            control_image, width=width, height=height,
            batch_size=batch_size * num_images_per_prompt,
            device=device, dtype=self.vae.dtype,
        )
        if seed_pixels.ndim == 4:
            seed_latents = self.vae.encode(seed_pixels).latent_dist.sample(generator=generator)
            seed_latents = (seed_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            h_s, w_s = seed_latents.shape[2:]
            packed_seed = self._pack_latents(
                seed_latents,
                batch_size * num_images_per_prompt,
                self.vae_latent_channels,
                h_s, w_s,
            )
        else:
            packed_seed = seed_pixels  # already packed

        # --- 3. Sample noise latents ---
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.vae_latent_channels,
            height, width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # --- 4. Set up the flow-matching schedule (resolution-aware shift) ---
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        mu = _calculate_shift(
            latents.shape[1],
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # --- 5. (Optionally) embedded-guidance vector for guidance-distilled FLUX ---
        guidance_vec = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance_vec = torch.full((latents.shape[0],), guidance_scale, device=device, dtype=torch.float32)

        # --- 6. Denoising loop ---
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Channel-wise concat of [noise || seed] in packed-token form (last dim).
                latent_in = torch.cat([latents, packed_seed], dim=2)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_in,
                    timestep=timestep / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    cb_out = callback_on_step_end(self, i, t, cb_kwargs) or {}
                    latents = cb_out.pop("latents", latents)
                    prompt_embeds = cb_out.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # --- 7. Decode ---
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_styleflow_pipeline(
    pretrained_model_name_or_path: str,
    checkpoint: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cuda",
) -> StyleFlowPipeline:
    """Load the FLUX backbone, widen the x_embedder, optionally load
    StyleFlow LoRA weights, and return a ready-to-call :class:`StyleFlowPipeline`.

    ``checkpoint`` can be either:
      * a directory containing ``pytorch_lora_weights.safetensors`` (the
        standard layout produced by ``train.py``), or
      * a direct path to a ``.safetensors`` / ``.bin`` LoRA file.
    """
    import os
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype
    )
    expand_x_embedder_for_seed_concat(transformer)
    pipe = StyleFlowPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            pipe.load_lora_weights(
                os.path.dirname(checkpoint) or ".",
                weight_name=os.path.basename(checkpoint),
            )
        else:
            pipe.load_lora_weights(checkpoint)
    pipe.to(device)
    return pipe
