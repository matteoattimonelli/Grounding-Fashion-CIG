import torch
import os
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import FluxControlPipeline
import torchvision.transforms as transforms
import argparse
from PIL import Image
import pandas as pd
from model import DiFashion, MutualEncoder
from diffusers import UNet2DConditionModel

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger

class ImageDataset(Dataset):
    def __init__(
            self, 
            root_path: str,
            dataset_name: str,
            transform: transforms,
            mode: str
            ) -> None:

        assert mode in ['train', 'valid', 'test']


        if dataset_name == 'fashionvc':
            train_file = 'train.csv'
            valid_file = 'valid.csv'
            test_file = 'test.csv'
        elif dataset_name == 'fashiontaobaoTB':
            train_file = 'train.csv'
            valid_file = 'valid.csv'
            test_file = 'test.csv'
        elif dataset_name == 'expreduced':
            train_file = 'train.csv'
            valid_file = 'valid.csv'
            test_file = 'test.csv'
        else:
            raise AssertionError("dataset_name must be ['fashionvc', 'expreduced', 'fashiontaobaoTB']!")

        if mode == 'train':
            f = os.path.join(root_path, 'files', train_file)
        elif mode == 'valid':
            f = os.path.join(root_path, 'files', valid_file)
        elif mode == 'test':
            f = os.path.join(root_path, 'files', test_file)
        
        self.data = pd.read_csv(f)
        self.transform = transforms.Compose(transform)
        self.root_path = root_path

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        prompt = data['bottom_description']
        top = self.transform(Image.open(os.path.join(self.root_path, 'img', f"{data['tshirt']}.jpg")).convert('RGB'))

        inputs = {
            "prompt": prompt,
            "bottom_id": data['positive_pant'],
            "top_id": data['tshirt'],
            "top": top
            }
        
        return inputs


if __name__ == '__main__':

    config = argparse.ArgumentParser(description="StyleFlow Generations.")
    # config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], required=True)
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], default='fashiontaobaoTB')
    # config.add_argument('--mode', choices=['train', 'valid', 'test'], required=True)
    config.add_argument('--mode', choices=['train', 'valid', 'test'], default='test')
    config.add_argument('--device', type=str, help="device", default='cuda')
    config.add_argument('--img_size', type=int, help="image size", default=512)
    config.add_argument('--batch_size', type=int, help="batch size", default=1)
    config.add_argument('--num_workers', type=int, help="num workers", default=multiprocessing.cpu_count())
    config.add_argument('--save_dir', type=str, help="save path", default=os.path.join(os.getcwd(), 'DiFashion-Gen-BASE'))
    config.add_argument('--weights_dir', type=str, help="weights_dir path", default=os.path.join(os.getcwd(), 'DiFashion-Gen-BASE', 'weights'))

    config.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./baselines/DiFashion/checkpoint-20000",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    config.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    config.add_argument(
        "--data_path",
        type=str,
        default='./data/',
        help="A folder containing the dataset for training and inference."
    )
    config.add_argument(
        '--img_folder_path',
        type=str,
        default='./data/polyvore/images'
    )
    config.add_argument(
        "--data_processed",
        type=bool,
        default=True,
        help="if the data is processed or not."
    )
    config.add_argument(
        "--dataset_name",
        type=str,
        default='polyvore',
        help="The name of the Dataset for training and inference."
    )
    config.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    config.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    config.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    config.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    config.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    config.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    config.add_argument(
        "--task",
        type=str,
        default="FITB",
        help="The task for evaluation: FITB or GOR (Generative Outfit Recommendation)."
    )
    config.add_argument(
        "--use_mutual_guidance",
        type=bool,
        default=True
    )
    # config.add_argument(
    #     "--use_history",
    #     type=bool,
    #     default=True
    # )

    config.add_argument(
        "--use_history",
        type=bool,
        default=False
    )

    config.add_argument(
        "--category_emb_size",
        type=int,
        default=64,
        help="Fashion item category embedding size.",
    )
    config.add_argument(
        "--hid_dim",
        type=int,
        default=256,
        help="Fashion encoder hidden dim."
    )
    config.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="The weight of mutual guidance."
    )
    config.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    config.add_argument(
        "--category_guidance_scale",
        type=float,
        default=12.0
    )
    config.add_argument(
        "--hist_guidance_scale",
        type=float,
        default=4.0
    )
    config.add_argument(
        "--mutual_guidance_scale",
        type=float,
        default=5.0
    )
    config.add_argument(
        "--train_batch_size", type=int, default=10, help="Batch size (per device) for the training dataloader."
    )
    config.add_argument("--num_train_epochs", type=int, default=100)
    config.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    config.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    config.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    config.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    config.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    config.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    config.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    config.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    config.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    config.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    config.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    config.add_argument("--use_ema_fashion", action="store_true", help="Whether to use EMA model for fashion encoder.")
    config.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    config.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    config.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    config.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    config.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    config.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    config.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    config.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    config.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    config.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    config.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    config.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    config.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    config.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest` to automatically select the last available checkpoint.'
        ),
    )
    config.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    config.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    config.add_argument(
        "--tracker_project_name",
        type=str,
        default="fashion_outfit_generation",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    config.add_argument("--run_name", type=str, default='', help="Run name")  

    args = config.parse_args()

    # logging_dir = args.logging_dir
    # accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.mixed_precision,
    #     log_with=args.report_to,
    #     project_config=accelerator_project_config,
    # )
    # device = accelerator.device

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    transforms_ = [
        transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    if args.dataset ==  'fashiontaobaoTB':
        # root_path = os.path.join(os.getcwd(), 'data', 'FashionTaobao-TB')
        root_path = os.path.join('./data', 'FashionTaobao-TB')
        save_dir = os.path.join(args.save_dir, 'FashionTaobao-TB', args.mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
    elif args.dataset ==  'fashionvc':
        # root_path = os.path.join(os.getcwd(), 'data', 'FashionVC')
        root_path = os.path.join('./data', 'FashionVC')
        save_dir = os.path.join(args.save_dir, 'FashionVC', args.mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    elif args.dataset ==  'expreduced':
        # root_path = os.path.join(os.getcwd(), 'data', 'ExpReduced')
        root_path = os.path.join('./data', 'ExpReduced')
        save_dir = os.path.join(args.save_dir, 'ExpReduced', args.mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    diffusion = DiFashion(args, 50, 'cuda').to(args.device)

    load_model = UNet2DConditionModel.from_pretrained('./baselines/DiFashion/checkpoint-15000', subfolder="unet", use_safetensors=False)
    diffusion.unet.register_to_config(**load_model.config)
    diffusion.unet.load_state_dict(load_model.state_dict())

    del load_model

    load_model = MutualEncoder.from_pretrained('./baselines/DiFashion/checkpoint-15000', subfolder="fashion_encoder", use_safetensors=False)
    diffusion.fashion_encoder.register_to_config(**load_model.config)
    diffusion.fashion_encoder.load_state_dict(load_model.state_dict())
    del load_model


    # diffusion.enable_model_cpu_offload() 
    diffusion.require_grads_ = False

    dataset = ImageDataset(
        root_path=root_path,
        dataset_name=args.dataset,
        transform=transforms_,
        mode=args.mode
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for batch in tqdm(dataloader):
        if args.batch_size == 1:
            null_img = torch.ones_like(batch['top'], device=args.device)
            template = diffusion.fashion_generation(
                prompt=batch['prompt'],
                top_img=batch['top'].to(args.device),
                height=args.img_size,
                width=args.img_size,
                num_inference_steps=args.num_inference_steps,
                category_guidance_scale=args.category_guidance_scale,
                hist_guidance_scale=args.hist_guidance_scale,
                mutual_guidance_scale=args.mutual_guidance_scale,
                null_img=null_img,
                generator=generator,
                return_dict=False
            )[0]
            # template = diffusion(
            #         batch['prompt'],
            #         control_image=batch['top'],
            #         height=args.img_size,
            #         width=args.img_size,
            #         guidance_scale=3.5,
            #         num_inference_steps=50,
            #         max_sequence_length=512,
            #         diffusion=torch.diffusion("cpu").manual_seed(0)
            #     ).images[0]
            
            template.save(os.path.join(save_dir, f"{batch['top_id'][0]}_{batch['bottom_id'][0]}_template.jpg"))
        else:
            null_img = torch.ones_like(batch['top'], device=args.device)
            template = diffusion.fashion_generation(
                prompt=batch['prompt'],
                top_img=batch['top'].to(args.device),
                height=args.img_size,
                width=args.img_size,
                num_inference_steps=args.num_inference_steps,
                category_guidance_scale=args.category_guidance_scale,
                hist_guidance_scale=args.hist_guidance_scale,
                mutual_guidance_scale=args.mutual_guidance_scale,
                null_img=null_img,
                generator=generator,
                return_dict=False
            )
            
            for i, t in enumerate(template):
                t.save(os.path.join(save_dir, f"{batch['top_id'][i]}_{batch['bottom_id'][i]}_template.jpg"))
        


