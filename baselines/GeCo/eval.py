import os
import random
from tqdm import tqdm
from model import GeCo
from utils.dataset import FashionDataset, FashionTaobaoTBDataset
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from baselines.custom_gan_text.models.generator import Generator
from itertools import product
import warnings
import argparse
import multiprocessing
from torchmetrics.functional.retrieval import retrieval_normalized_dcg

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True


def eval_fn(
        args
    ):


    num_workers = args.num_workers
    emb_dim = args.emb_dim
    img_size = args.img_size
    dataset = args.dataset
    

    # weight_path = "./baselines/GeCo/weights/geco_fashiontaobaoTB_50_0.5_1.0_0.01_0.1.pt"
    # generator_path = './baselines/custom_gan_text/weights/generator_fashiontaobaoTB.pt'

    # weight_path = "./baselines/GeCo/weights/geco_expreduced_1_0.5_0.5_0.01_0.1.pt"
    # generator_path = './baselines/custom_gan_text/weights/generator_expreduced.pt'

    weight_path = "./baselines/GeCo/weights/geco_fashionvc_1_0.5_0.5_0.01_0.1.pt"
    generator_path = './baselines/custom_gan_text/weights/generator_fashionvc.pt'

    transforms_ = [
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    path = os.path.join(os.getcwd(), 'data')

    if args.dataset == 'fashionvc':
        root_path = os.path.join(path, 'FashionVC')
        cir_dataset = FashionDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size, name=dataset)

    elif dataset == 'expreduced':
        root_path = os.path.join(path, 'ExpReduced')
        cir_dataset = FashionDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size, name=dataset)

    else:
        root_path = os.path.join(path, 'FashionTaobao-TB')
        cir_dataset = FashionTaobaoTBDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size)

    geco = GeCo(
        emb_dim=emb_dim,
        learning_rate=1e-4,
        alpha=0,
        beta=0,
        gamma=0,
        temperature=0
    ).to(device)

    geco.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'])
    geco.eval()

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device)['model_state_dict'])
    generator.eval()

    cir_dl = DataLoader(
            cir_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    with torch.no_grad():
        mrr10, mrr50, r10, r50, n10, n50 = compute_metrics(cir_dl, generator, geco)

    return mrr10, mrr50, r10, r50, n10, n50


@torch.no_grad()
def compute_metrics(
    cir_dl, 
    generator, 
    compatibility_network
) -> np.array:
    
    test_csv = cir_dl.dataset.data
    test_loop = tqdm(test_csv.iterrows(), desc='Computing metrics...', leave=False, total=len(test_csv))

    tshirt_dir = cir_dl.dataset.img_dir
    pants_dir = cir_dl.dataset.img_dir

    transform = transforms.Compose([
            transforms.Resize((128, 128), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    unique_pants = list(test_csv['positive_pant'].unique())

    negative_encs = []

    for neg in tqdm(unique_pants, desc='Preparing negatives...'):
        negative_pant = transform(Image.open(os.path.join(pants_dir, str(neg) + '.jpg')).convert('RGB')).to(
            device)
        negative_encs.append(compatibility_network.forward_bottom(negative_pant[None]).cpu().numpy())

    negative_encs = list(negative_encs)

    running_mrr_10 = []
    running_mrr_50 = []
    running_recall_10 = []
    running_recall_50 = []
    running_ndcg_10 = []
    running_ndcg_50 = []

    for _, batch in test_loop:
        tshirt = transform(Image.open(os.path.join(tshirt_dir, str(batch['tshirt']) + '.jpg')).convert('RGB')).to(
            device)
        fake = generator(tshirt[None])        
        query_enc = compatibility_network.forward_query(tshirt[None], fake)
        true_pant = transform(
            Image.open(os.path.join(pants_dir, f"{str(batch['positive_pant'])}.jpg")).convert('RGB')).to(
            device)
        true_enc = compatibility_network.forward_bottom(true_pant[None])
        idx = unique_pants.index(batch['positive_pant'])

        actual_neg = torch.tensor(negative_encs[:idx] + negative_encs[idx + 1:]).to(device)

        all_enc = torch.vstack([true_enc[None], actual_neg]).squeeze()

        similarity_matrix = torch.mul(query_enc, all_enc).sum(dim=1)

        _, indices = similarity_matrix.topk(k=all_enc.shape[0], largest=True)

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

        labels = torch.tensor([1 if i == 0 else 0 for i in range(len(similarity_matrix))], device=similarity_matrix.device)
        running_ndcg_10.append(retrieval_normalized_dcg(similarity_matrix, labels, top_k=10).cpu().numpy())
        running_ndcg_50.append(retrieval_normalized_dcg(similarity_matrix, labels, top_k=50).cpu().numpy())


    return np.mean(running_mrr_10), np.mean(running_mrr_50), np.mean(running_recall_10), np.mean(running_recall_50), np.mean(running_ndcg_10), np.mean(running_ndcg_50)


if __name__ == '__main__':
    config = argparse.ArgumentParser(description="Training and evaluation script for GeCo.")
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], required=True)
    config.add_argument('--emb_dim', type=int, help="Embedding dimension.", default=128)
    config.add_argument('--device', type=str, help="device", default='cuda')
    config.add_argument('--img_size', type=int, help="image size", default=128)
    config.add_argument('--num_workers', type=int, help="num workers", default=multiprocessing.cpu_count())

    args = config.parse_args()

    
    mrr10, mrr50, r10, r50, n10, n50 = eval_fn(args)
    print(mrr10, mrr50, r10, r50, n10, n50)
        
