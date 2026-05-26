import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import FashionDataset, FashionTaobaoTBDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import argparse
import multiprocessing
import random
import pandas as pd
import utils
from transformers import CLIPTokenizer, CLIPTextModel



random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True

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
        bottom = self.transform(Image.open(os.path.join(self.root_path, 'img', f"{data['positive_pant']}.jpg")).convert('RGB'))

        inputs = {
            "prompt": prompt,
            "bottom_id": data['positive_pant'],
            "top_id": data['tshirt'],
            "top": top,
            "bottom": bottom
            }
        
        return inputs


def weights_init(m) -> None:
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_step(
        gen: Generator,
        disc: Discriminator,
        train_dl: DataLoader,
        gen_optimizer: torch.optim, 
        disc_optimizer: torch.optim, 
        L1_Loss: torch.nn, 
        BCE_Loss: torch.nn, 
        L1_LAMBDA:int,
        device: torch.device
    ) -> None:

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)


    
    loop = tqdm(train_dl, desc='Training')
    running_g, running_g_l1, running_g_bce = [], [], []
    running_d = []


    for batch in loop:
        x = batch["top"].to(device)
        y = batch["bottom"].to(device)
        text = batch["prompt"]

        t_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        t_emb = text_model(**t_inputs).pooler_output
        
        # Train Discriminator

        y_fake = gen(x, t_emb)

        D_real = disc(x, y, t_emb.detach())

        D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach(), t_emb.detach())

        D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2

        running_d.append(D_loss.detach().cpu())

        disc_optimizer.zero_grad()
        D_loss.backward()
        disc_optimizer.step()
        
        # Train Generator
        
        D_fake = disc(x, y_fake, t_emb.detach())
        G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
        L1 = L1_Loss(y_fake,y) * L1_LAMBDA
        G_loss = G_fake_loss + L1

        running_g.append(G_loss.detach().cpu())
        running_g_l1.append(L1.detach().cpu())
        running_g_bce.append(G_fake_loss.detach().cpu())
        
        gen_optimizer.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        logs = {
            'G loss': np.mean(running_g), 
            "G L1": np.mean(running_g_l1), 
            "G BCE": np.mean(running_g_bce), 
            "D loss": np.mean(running_d)
        }
        
        loop.set_postfix(**logs)


def train(config):

    if not os.path.exists(config.weights_dir):
        os.mkdir(config.weights_dir)

    if config.save_some_samples is not None:
        if not os.path.exists(config.save_some_samples):
            os.makedirs(config.save_some_samples)

    generator_path = os.path.join(config.weights_dir, f'generator_{config.dataset}.pt')
    discriminator_path = os.path.join(config.weights_dir, f'discriminator_{config.dataset}.pt')

    device = torch.device(config.device)
    gen = Generator().to(device)
    gen.apply(weights_init)

    disc = Discriminator(in_channels=3).to(device)


    disc_optimizer = torch.optim.Adam(
        disc.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, 0.999)
    )
    
    gen_optimizer = torch.optim.Adam(
        gen.parameters(),
        lr=config.learning_rate, 
        betas=(config.beta1, 0.999)
    )
    
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()
    

    transforms_ = [
        transforms.Resize((config.img_size, config.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    path = os.path.join(os.getcwd(), 'data')
    
    if args.dataset == 'fashionvc':
        root_path = os.path.join(path, 'FashionVC')
        # train_dataset = FashionDataset(root_path, mode='train', transforms_=img_trans, img_size=args.img_size, name=args.dataset)
        # valid_dataset = FashionDataset(root_path, mode='valid', transforms_=img_trans, img_size=args.img_size, name=args.dataset)

        train_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='train'
            )
        
        valid_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='valid'
            )

    elif args.dataset == 'expreduced':
        root_path = os.path.join(path, 'ExpReduced')
        # train_dataset = FashionDataset(root_path, mode='train', transforms_=img_trans, img_size=args.img_size, name=args.dataset)
        # valid_dataset = FashionDataset(root_path, mode='valid', transforms_=img_trans, img_size=args.img_size, name=args.dataset)
        train_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='train'
            )
        
        valid_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='valid'
            )

    elif args.dataset == 'fashiontaobaoTB':
        root_path = os.path.join(path, 'FashionTaobao-TB')
        # train_dataset = FashionTaobaoTBDataset(root_path, mode='train', transforms_=img_trans, img_size=args.img_size)
        # valid_dataset = FashionTaobaoTBDataset(root_path, mode='valid', transforms_=img_trans, img_size=args.img_size)
        train_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='train'
            )
        
        valid_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='valid'
            )
    else:
        raise ValueError("Dataset must be in ['fashionvc', 'expreduced', 'fashiontaobaoTB']")

    train_dl = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_dl = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    
    for epoch in tqdm(range(config.num_epochs), desc='Epochs'):
        train_step(
            gen, 
            disc, 
            train_dl, 
            gen_optimizer, 
            disc_optimizer, 
            L1_Loss,
            BCE_Loss, 
            L1_LAMBDA=config.L1Lambda, 
            device=device
        )

        if epoch%2==0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': gen.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
                'loss': BCE_Loss,
            }, generator_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': disc.state_dict(),
                'optimizer_state_dict': disc_optimizer.state_dict(),
                'loss': BCE_Loss,
            }, discriminator_path)

        if epoch % 2 == 0 and config.save_some_samples is not None:
            utils.save_some_examples(
                gen, 
                val_dl, 
                epoch, 
                folder=config.save_some_samples, 
                device=device
            )


if __name__ == '__main__':
    config = argparse.ArgumentParser(description="Training script for CIGM.")

    config.add_argument('--num_epochs', type=int, help="Number of epochs.", default=200)
    config.add_argument('--beta1', type=float, help="Adam Beta1.", default=0.5)
    config.add_argument('--learning_rate', type=float, help="Learning rate", default=2e-4)
    config.add_argument('--L1Lambda', type=int, help="L1Lambda", default=100)
    config.add_argument('--device', type=str, help="device", default='cuda')
    config.add_argument('--img_size', type=int, help="image size", default=128)
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'])
    config.add_argument('--train_batch_size', type=int, help="train batch size", default=64)
    config.add_argument('--valid_batch_size', type=int, help="valid batch size", default=16)
    config.add_argument('--num_workers', type=int, help="num workers", default=multiprocessing.cpu_count())
    config.add_argument('--weights_dir', type=str, help="weights path", default=os.path.join(os.getcwd(), 'baselines', 'CIGM_text', 'weights'))
    config.add_argument('--save_some_samples', type=str, help="path for saving samples", default=None)

    args = config.parse_args()
    train(args)
   