import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model.generator import Generator
from model.discriminator import Discriminator
from model.bpr import BPRNet
from model.encoder import PantsEncoder
from utils.dataset import FashionDataset, FashionTaobaoTBDataset
from torchvision.utils import save_image
from itertools import product
import multiprocessing
import warnings
from torchmetrics.functional.retrieval import retrieval_normalized_dcg
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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
        self.img_dir = os.path.join(self.root_path, 'img')

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        prompt = data['low']
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
    


def save_some_examples(
        gen,
        val_loader: DataLoader,
        epoch: int,
        folder: str,
        device: torch.device
) -> None:
    batch = next(iter(val_loader))
    x = batch["top"]
    y = batch["bottom"]

    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        _, _, y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        if epoch == 0:
            save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()



def eval_fn(
        alpha, 
        beta, 
        args
    ):
    num_workers = args.num_workers
    img_size = args.img_size
    epochs = args.epochs
    dataset = args.dataset


    transforms_ = [
        transforms.Resize((64, 64), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    path = os.path.join(os.getcwd(), 'data')
    if args.dataset == 'fashionvc':
        root_path = os.path.join(path, 'FashionVC')
        cir_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='test'
            )

    elif dataset == 'expreduced':
        root_path = os.path.join(path, 'ExpReduced')
        cir_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='test'
            )

    else:
        root_path = os.path.join(path, 'FashionTaobao-TB')
        cir_dataset = ImageDataset(
            root_path=root_path,
            dataset_name=args.dataset,
            transform=transforms_,
            mode='test'
            )

    
    generator = Generator(conv_filters=[64, 128, 256, 512, 512, 512]).to(device)
    generator.load_state_dict(torch.load(os.path.join(args.weights_dir, f'mgcm_generator_{dataset}.pt'))['model_state_dict'])
    generator.eval()

    encoder = PantsEncoder().to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.weights_dir, f'mgcm_encoder_{dataset}.pt'))['model_state_dict'])
    encoder.eval()

    bpr_net = BPRNet().to(device)
    bpr_net.load_state_dict(torch.load(os.path.join(args.weights_dir, f'mgcm_bpr_{dataset}.pt'))['model_state_dict'])
    bpr_net.eval()

    cir_dl = DataLoader(
            cir_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    with torch.no_grad():
        mrr_10, mrr50, r10, r50, n10, n50 = compute_metrics(cir_dl, generator, encoder, bpr_net, alpha, beta)

    del generator, encoder, bpr_net
    torch.cuda.empty_cache()

    return mrr_10, mrr50, r10, r50, n10, n50


@torch.no_grad()
def compute_metrics(cir_dl, generator, pants_encoder, bpr_net, alpha, beta) -> np.array:
    bpr_net._set_metric(metric='MRR')
    test_csv = cir_dl.dataset.data
    test_loop = tqdm(cir_dl, desc='Computing metrics...', leave=False)

    if isinstance(cir_dl.dataset, FashionTaobaoTBDataset):
        tshirt_dir = cir_dl.dataset.img_dir
        pants_dir = cir_dl.dataset.img_dir
    else:
        tshirt_dir = cir_dl.dataset.img_dir
        pants_dir = cir_dl.dataset.img_dir

    transform = transforms.Compose([
        transforms.Resize((64, 64), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    unique_pants = list(test_csv['positive_pant'].unique())

    negative_encs = []

    for neg in tqdm(unique_pants, desc='Preparing negatives...'):
        negative_pant = transform(Image.open(os.path.join(pants_dir, str(neg) + '.jpg')).convert('RGB')).to(device)
        emb_neg = pants_encoder(negative_pant[None]).squeeze()
        negative_encs.append(emb_neg.cpu().numpy())

    negative_encs = list(negative_encs)

    running_recall_10 = []
    running_recall_50 = []
    running_ndcg_10 = []
    running_ndcg_50 = []
    running_mrr_10 = []
    running_mrr_50 = []

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)


    for batch in test_loop:
        tshirt = batch['top'].to(device)
        true_pant = batch['bottom'].to(device)

        text = batch["prompt"]

        t_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        t_emb = text_model(**t_inputs).pooler_output

        enc_A, enc_Bg, _ = generator(tshirt, t_emb)

        enc_B = pants_encoder(true_pant)
        idx = unique_pants.index(batch['bottom_id'][0])

        actual_neg = torch.tensor(negative_encs[:idx] + negative_encs[idx + 1:]).to(device)

        enc_B_tot = torch.vstack([enc_B, actual_neg])

        it_sim_ij = torch.sum(torch.abs(enc_Bg - enc_B_tot), dim=(1, 2, 3))


        enc_A, enc_B, enc_B1 = bpr_net(enc_A, enc_B, actual_neg)

        enc_B_tot = torch.cat([enc_B, enc_B1])

        ii_sim_ij_v = torch.sum(enc_A * enc_B_tot, dim=(1))  # [B]

        mij = beta * it_sim_ij + alpha * ii_sim_ij_v

        _, indices = mij.topk(k=enc_B_tot.shape[0], largest=True)

        indices = list(indices)

        if 0 in indices[:10]:
            mrr_10 = 1 / (indices.index(0) + 1)
        else:
            mrr_10 = 0

        running_mrr_10.append(mrr_10)

        if 0 in indices[:50]:
            mrr_50 = 1 / (indices.index(0) + 1)
        else:
            mrr_50 = 0

        running_mrr_50.append(mrr_50)
        
        # Compute Recall@10
        recall_10 = 1 if 0 in indices[:10] else 0
        running_recall_10.append(recall_10)

        # Compute Recall@50
        recall_50 = 1 if 0 in indices[:50] else 0
        running_recall_50.append(recall_50)

        labels = torch.tensor([1 if i == 0 else 0 for i in range(len(mij))], device=mij.device)
        running_ndcg_10.append(retrieval_normalized_dcg(mij, labels, top_k=10).cpu().numpy())
        running_ndcg_50.append(retrieval_normalized_dcg(mij, labels, top_k=50).cpu().numpy())


    return np.mean(running_mrr_10), np.mean(running_mrr_50), np.mean(running_recall_10), np.mean(running_recall_50), np.mean(running_ndcg_10), np.mean(running_ndcg_50)


if __name__ == '__main__':
    config = argparse.ArgumentParser(description="Training and evaluation script for MGCM.")
    config.add_argument('--alpha_values', nargs='+', type=float, help='List of alpha values', default=1)
    config.add_argument('--beta_values', nargs='+', type=float, help='List of beta values', default=0.01)
    config.add_argument('--mi_values', nargs='+', type=float, help='List of mi values', default=0.1)
    config.add_argument('--ni_values', nargs='+', type=float, help='List of ni values', default=0.01)
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], required=True)
    config.add_argument('--epochs', type=int, help="Number of epochs.", default=60)
    config.add_argument('--learning_rate', type=float, help="Learning rate.", default=0.0002)
    config.add_argument('--device', type=str, help="device.", default='cuda')
    config.add_argument('--batch_size', type=int, help="Training Batch Size.", default=420)
    config.add_argument('--valid_batch_size', type=int, help="Valid Batch Size.", default=16)
    config.add_argument('--num_workers', type=int, help="Dataloader workers.", default=multiprocessing.cpu_count())
    config.add_argument('--img_size', type=int, help="Image Size.", default=64)
    config.add_argument('--save_some_samples', type=str, help="Path where to save some batch images.", default=None)
    config.add_argument('--out_csv', type=str, help="path of the output csv", default=os.path.join(os.getcwd(), 'baselines', 'MGCM_text', 'out_vc.csv'))
    config.add_argument('--weights_dir', type=str, help="weights path", default=os.path.join(os.getcwd(), 'baselines', 'MGCM_text', 'weights'))

    args = config.parse_args()

    alphas = args.alpha_values if type(args.alpha_values) is list else [args.alpha_values]
    betas = args.beta_values if type(args.beta_values) is list else [args.beta_values]
    mis = args.mi_values if type(args.mi_values) is list else [args.mi_values]
    nis = args.ni_values if type(args.ni_values) is list else [args.ni_values]

    hyperparameter_grid = list(product(alphas, betas, mis, nis))
    hyperparameter_grid = [quadruple for quadruple in hyperparameter_grid]



    mrr10, mrr50, r10, r50, n10, n50 = eval_fn(1, 0.01, args)
    print(mrr10, mrr50, r10, r50, n10, n50)
    
        
