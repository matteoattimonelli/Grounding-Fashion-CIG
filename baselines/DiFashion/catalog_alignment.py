import os
import random
from tqdm import tqdm
from utils.dataset import FTaobaoTBFMTemplateDataset, FashionFMTemplateDataset
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from itertools import product
import warnings
import argparse
import multiprocessing
from torchmetrics.functional.retrieval import retrieval_normalized_dcg
import open_clip
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

        self.img_dir = os.path.join(root_path, 'img')
        
        self.data = pd.read_csv(f)
        self.transform = transforms.Compose(transform)
        self.root_path = root_path

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        prompt = data['bottom_description']
        positive_pant = self.transform(Image.open(os.path.join(self.root_path, 'img', f"{data['positive_pant']}.jpg")).convert('RGB'))

        inputs = {
            "prompt": prompt,
            "bottom_id": data['positive_pant'],
            "top_id": data['tshirt'],
            "positive_pant": positive_pant
            }
        
        return inputs


def eval_fn(args):    
    transforms_ = [
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    path = os.path.join(os.getcwd(), 'data')
    # path = './data'

    if args.dataset == 'fashionvc':
        root_path = os.path.join(path, 'FashionVC')
        cir_dataset = ImageDataset(root_path, dataset_name='fashionvc', transform=transforms_, mode='test')
        # cir_dataset = FashionFMTemplateDataset(root_path, dataset_name = 'fashionvc', temp_path=os.path.join(os.getcwd(), 'StyleFlowCIR', 'FashionVC'), mode='CIR', transforms_=transforms_, img_size=224)

    elif args.dataset == 'expreduced':
        root_path = os.path.join(path, 'ExpReduced')
        cir_dataset = ImageDataset(root_path, dataset_name='expreduced', transform=transforms_, mode='test')
        # cir_dataset = FashionFMTemplateDataset(root_path, dataset_name='expreduced', temp_path=os.path.join(os.getcwd(), 'StyleFlowCIR', 'ExpReduced'), mode='CIR', transforms_=transforms_, img_size=224)

    else:
        root_path = os.path.join(path, 'FashionTaobao-TB')
        cir_dataset = ImageDataset(root_path, dataset_name='fashiontaobaoTB', transform=transforms_, mode='test')
        # cir_dataset = FTaobaoTBFMTemplateDataset(root_path, temp_path=os.path.join(os.getcwd(), 'StyleFlowCIR', 'FashionTaobao-TB'), mode='CIR', transforms_=transforms_, img_size=224)

    model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
    model = model.to(args.device)
    model.eval()

    cir_dl = DataLoader(
        cir_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
        )

    with torch.no_grad():
       mrr10, mrr50, r10, r50, n10, n50 = compute_metrics(cir_dl, model)

    return mrr10, mrr50, r10, r50, n10, n50



@torch.no_grad()
def compute_metrics(
    cir_dl, 
    network
) -> np.array:
    
    test_csv = cir_dl.dataset.data
    test_loop = tqdm(test_csv.iterrows(), desc='Computing metrics...', leave=False, total=len(test_csv))

    pants_dir = cir_dl.dataset.img_dir

    transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    unique_pants = list(test_csv['positive_pant'].unique())

    negative_encs = []

    for neg in tqdm(unique_pants, desc='Preparing negatives...'):
        negative_pant = transform(Image.open(os.path.join(pants_dir, str(neg) + '.jpg')).convert('RGB')).to(
            device)
        neg_enc = network.encode_image(negative_pant[None])
        neg_enc = neg_enc / neg_enc.norm(p=2, dim=-1, keepdim=True)
        negative_encs.append(neg_enc.cpu().numpy())

    negative_encs = list(negative_encs)

    running_mrr_10 = []
    running_mrr_50 = []
    running_recall_10 = []
    running_recall_50 = []
    running_ndcg_10 = []
    running_ndcg_50 = []

    for _, batch in test_loop:

        # true_pant = transform(
        #     Image.open(os.path.join(pants_dir, f"{str(batch['positive_pant'])}.jpg")).convert('RGB')).to(
        #     device)
        true_pant = transform(
            Image.open(os.path.join('./baselines/DiFashion/outputs/ExpReduced/test/prompt_empty', f"{str(batch['tshirt'])}_{str(batch['positive_pant'])}_template.jpg")).convert('RGB')).to(
            device)
        
        true_enc = network.encode_image(true_pant[None])
        true_enc = true_enc / true_enc.norm(p=2, dim=-1, keepdim=True)

        idx = unique_pants.index(batch['positive_pant'])

        actual_neg = torch.tensor(negative_encs[:idx] + negative_encs[idx + 1:]).squeeze().to(device)

        all_enc = torch.vstack([torch.tensor(negative_encs[idx]).to(device), actual_neg]).squeeze()

        similarity_matrix = torch.mul(true_enc, all_enc).sum(dim=1)

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
    config = argparse.ArgumentParser(description="Catalog-alignment retrieval evaluation script.")
    # config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], required=True)
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], default='fashionvc')
    config.add_argument('--device', type=str, help="device", default='cuda')
    config.add_argument('--num_workers', type=int, help="num workers", default=multiprocessing.cpu_count())

    args = config.parse_args()

    
    mrr10, mrr50, r10, r50, n10, n50 = eval_fn(args)

    print(mrr10, mrr50, r10, r50, n10, n50)

        



