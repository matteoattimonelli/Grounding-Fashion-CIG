import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.generator import GeneratorSkip
from model.discriminator import Discriminator
from model.bpr import BPRNet
from model.encoder import PantsEncoder
from utils.dataset import FashionDataset, FashionTaobaoTBDataset
from torchvision.utils import save_image
from itertools import product
import multiprocessing
import warnings
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

def save_some_examples(
        gen,
        val_loader: DataLoader,
        epoch: int,
        folder: str,
        device: torch.device
) -> None:
    batch = next(iter(val_loader))
    x = batch["tshirt_image"]
    y = batch["gt_image"]

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


def train_step(
        generator: GeneratorSkip,
        discriminator: Discriminator,
        encoder: PantsEncoder,
        bpr_net: BPRNet,
        generator_opt: torch.optim,
        discriminator_opt: torch.optim,
        bpr_opt: torch.optim,
        generator_loss: torch.nn,
        pixel_loss: torch.nn,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epoch: int,
        device: torch.device,
        img_size: int = 64,
        eps: float = 1e-12, 
        alpha: float = None,
        beta: float = None,
        mi: float = None,
        ni: float = None
) -> None:

    generator.train()
    discriminator.train()
    encoder.train()
    bpr_net.train()

    loop = tqdm(train_dl, desc='Training')
    val_loop = tqdm(val_dl, desc='Validation')
    running_auc, running_g_loss_final, running_d_loss, running_pixel_loss, running_bpr_loss = [], [], [], [], []

    running_auc_val, running_g_loss_final_val, running_d_loss_val, running_pixel_loss_val, running_bpr_loss_val = [], [], [], [], []

    for batch in loop:
        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = torch.stack([true_pants.roll(shifts=i + 1, dims=0) for i in range(3)], dim=1)

        enc_A, enc_Bg, fake_B = generator(tshirts)
        real_B = discriminator(tshirts, true_pants)

        fake_rec_B = discriminator(tshirts, fake_B)
        d_loss = 0.5 * generator_loss(fake_rec_B.detach(),
                                      torch.zeros_like(fake_rec_B.detach()) + 0.5 * generator_loss(real_B,
                                                                                                   torch.ones_like(
                                                                                                       real_B)))
        d_loss_total = ni * d_loss

        discriminator_opt.zero_grad()
        d_loss_total.backward()
        discriminator_opt.step()

        enc_A, enc_Bg, fake_B = generator(tshirts)
        fake_rec_B = discriminator(tshirts, fake_B)

        g_loss = 0.5 * generator_loss(fake_rec_B, torch.ones_like(fake_rec_B))
        p_loss = pixel_loss(fake_B, true_pants)

        g_loss_total = mi * g_loss + 10000 * p_loss

        generator_opt.zero_grad()
        g_loss_total.backward()
        generator_opt.step()

        generator.eval()
        enc_A, enc_Bg, fake_B = generator(tshirts)

        pant_enc_B = encoder(true_pants)
        pant_enc_B1 = encoder(negative_pants.view(-1, 3, img_size, img_size)).view(tshirts.shape[0], 3, 512, pant_enc_B.shape[-1], pant_enc_B.shape[-1])

        it_sim_ij = torch.sum(torch.abs(enc_Bg - pant_enc_B), dim=(1, 2, 3))
        it_sim_ik = torch.sum(torch.abs(enc_Bg.unsqueeze(1).repeat(1, 3, 1, 1, 1) - pant_enc_B1), dim=(2, 3, 4))

        enc_A, enc_B, enc_B1 = bpr_net(enc_A, pant_enc_B, pant_enc_B1)

        ii_sim_ij_v = torch.sum(enc_A * enc_B, dim=1)  # [B]
        ii_sim_ik_v = torch.sum(enc_A.unsqueeze(1).repeat(1, 3, 1) * enc_B1, dim=(2))  # [B]

        mij = beta * it_sim_ij + alpha * ii_sim_ij_v
        mik = beta * it_sim_ik + alpha * ii_sim_ik_v

        score = torch.subtract(mij.unsqueeze(1).repeat(1, 3), mik)

        auc = (mij.unsqueeze(1).repeat(1, 3) > mik).bool().sum().item() / (score.shape[0] * 3)
        running_auc.append(auc)

        bpr_loss = -torch.mean(torch.log(torch.sigmoid(score) + torch.tensor(float(eps))))

        bpr_opt.zero_grad()
        bpr_loss.backward()
        bpr_opt.step()

        running_g_loss_final.append(g_loss_total.item())
        running_d_loss.append(d_loss_total.item())
        running_bpr_loss.append(bpr_loss.item())
        running_pixel_loss.append(p_loss.item())

        generator.train()

        logs = {
            "G loss": np.mean(running_g_loss_final),
            "L1": np.mean(running_pixel_loss),
            "BPR Loss": np.mean(running_bpr_loss),
            "D loss": np.mean(running_d_loss),
            "epoch": epoch,
            'auc': np.mean(running_auc)
        }

        loop.set_postfix(**logs)

    generator.eval()
    discriminator.eval()
    encoder.eval()
    bpr_net.eval()

    for batch in val_loop:
        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = batch['neg_images'].to(device)

        enc_A, enc_Bg, fake_B = generator(tshirts)
        real_B = discriminator(tshirts, true_pants)

        fake_rec_B = discriminator(tshirts, fake_B)
        d_loss = 0.5 * generator_loss(fake_rec_B.detach(),
                                      torch.zeros_like(fake_rec_B.detach()) + 0.5 * generator_loss(real_B, torch.ones_like(real_B)))
        d_loss_total = ni * d_loss

        enc_A, enc_Bg, fake_B = generator(tshirts)
        fake_rec_B = discriminator(tshirts, fake_B)

        g_loss = 0.5 * generator_loss(fake_rec_B, torch.ones_like(fake_rec_B))
        p_loss = pixel_loss(fake_B, true_pants)

        g_loss_total = mi * g_loss + 10000 * p_loss

        enc_A, enc_Bg, fake_B = generator(tshirts)

        pant_enc_B = encoder(true_pants)
        pant_enc_B1 = encoder(negative_pants.view(-1, 3, img_size, img_size)).view(tshirts.shape[0], 3, 512, pant_enc_B.shape[-1], pant_enc_B.shape[-1])

        it_sim_ij = torch.sum(torch.abs(enc_Bg - pant_enc_B), dim=(1, 2, 3))
        it_sim_ik = torch.sum(torch.abs(enc_Bg.unsqueeze(1).repeat(1, 3, 1, 1, 1) - pant_enc_B1), dim=(2, 3, 4))

        enc_A, enc_B, enc_B1 = bpr_net(enc_A, pant_enc_B, pant_enc_B1)

        ii_sim_ij_v = torch.sum(enc_A * enc_B, dim=1)  # [B]
        ii_sim_ik_v = torch.sum(enc_A.unsqueeze(1).repeat(1, 3, 1) * enc_B1, dim=(2))  # [B]

        mij = beta * it_sim_ij + alpha * ii_sim_ij_v
        mik = beta * it_sim_ik + alpha * ii_sim_ik_v

        score = torch.subtract(mij.unsqueeze(1).repeat(1, 3), mik)

        auc = (mij.unsqueeze(1).repeat(1, 3) > mik).bool().sum().item() / (score.shape[0] * 3)

        running_auc_val.append(auc)

        bpr_loss = -torch.mean(torch.log(torch.sigmoid(score) + torch.tensor(float(eps))))

        running_g_loss_final_val.append(g_loss_total.item())
        running_d_loss_val.append(d_loss_total.item())
        running_bpr_loss_val.append(bpr_loss.item())
        running_pixel_loss_val.append(p_loss.item())

        val_logs = {
            "G loss VAL": np.mean(running_g_loss_final_val),
            "L1 VAL": np.mean(running_pixel_loss_val),
            "BPR Loss VAL": np.mean(running_bpr_loss_val),
            "D loss VAL": np.mean(running_d_loss_val),
            "epoch": epoch,
            'auc VAL': np.mean(running_auc_val)
        }

        val_loop.set_postfix(**val_logs)


def train(
        alpha, 
        beta, 
        mi,
        ni, 
        args
    ):
    batch_size = args.batch_size
    valid_batch_size = args.valid_batch_size
    num_workers = args.num_workers
    img_size = args.img_size
    epochs = args.epochs
    dataset = args.dataset

    if not os.path.exists(args.weights_dir):
        os.mkdir(args.weights_dir)

    if args.save_some_samples is not None:
        if not os.path.exists(args.save_some_samples):
            os.makedirs(args.save_some_samples)

    transforms_ = [
        transforms.Resize((64, 64), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    path = os.path.join(os.getcwd(), 'data')
    if args.dataset == 'fashionvc':
        root_path = os.path.join(path, 'FashionVC')
        train_dataset = FashionDataset(root_path, mode='train', transforms_=transforms_, img_size=img_size, name=dataset)
        valid_dataset = FashionDataset(root_path, mode='valid', transforms_=transforms_, img_size=img_size, name=dataset)
        test_dataset = FashionDataset(root_path, mode='test', transforms_=transforms_, img_size=img_size, name=dataset)
        cir_dataset = FashionDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size, name=dataset)

    elif dataset == 'expreduced':
        root_path = os.path.join(path, 'ExpReduced')
        train_dataset = FashionDataset(root_path, mode='train', transforms_=transforms_, img_size=img_size, name=dataset)
        valid_dataset = FashionDataset(root_path, mode='valid', transforms_=transforms_, img_size=img_size, name=dataset)
        test_dataset = FashionDataset(root_path, mode='test', transforms_=transforms_, img_size=img_size, name=dataset)
        cir_dataset = FashionDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size, name=dataset)

    else:
        root_path = os.path.join(path, 'FashionTaobao-TB')
        train_dataset = FashionTaobaoTBDataset(root_path, mode='train', transforms_=transforms_, img_size=img_size)
        valid_dataset = FashionTaobaoTBDataset(root_path, mode='valid', transforms_=transforms_, img_size=img_size)
        test_dataset = FashionTaobaoTBDataset(root_path, mode='test', transforms_=transforms_, img_size=img_size)
        cir_dataset = FashionTaobaoTBDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size)


    generator = GeneratorSkip(conv_filters=[64, 128, 256, 512, 512, 512]).to(device)
    discriminator = Discriminator().to(device)
    encoder = PantsEncoder().to(device)
    bpr_net = BPRNet().to(device)

    discriminator_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=0.01
    )

    generator_opt = torch.optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=0.01
    )

    parameters_to_optimize_bpr = list(encoder.parameters()) + list(bpr_net.parameters())

    bpr_opt = torch.optim.Adam(
        parameters_to_optimize_bpr,
        lr=args.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=0.01
    )

    generator_loss = nn.MSELoss()
    pixel_loss = nn.L1Loss()

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dl = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


    for epoch in tqdm(range(epochs), desc='Epochs'):
        train_step(
            generator=generator,
            discriminator=discriminator,
            encoder=encoder,
            bpr_net=bpr_net,
            generator_opt=generator_opt,
            discriminator_opt=discriminator_opt,
            bpr_opt=bpr_opt,
            generator_loss=generator_loss,
            pixel_loss=pixel_loss,
            train_dl=train_dl,
            val_dl=val_dl,
            epoch=epoch,
            device=device,
            img_size=img_size,
            eps=1e-12,
            alpha=alpha,
            beta=beta,
            mi=mi,
            ni=ni
        )

        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': generator_opt.state_dict(),
            }, os.path.join(args.weights_dir, f'pix2pixcm_generator_{dataset}.pt'))

            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': discriminator_opt.state_dict(),
            }, os.path.join(args.weights_dir, f'pix2pixcm_discriminator_{dataset}.pt'))

            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': encoder.state_dict(),
            }, os.path.join(args.weights_dir, f'pix2pixcm_encoder_{dataset}.pt'))

            torch.save({
                'epoch': epoch,
                'model_state_dict': bpr_net.state_dict(),
                'optimizer_state_dict': bpr_net.state_dict(),
            }, os.path.join(args.weights_dir, f'pix2pixcm_bpr_{dataset}.pt'))

            if args.save_some_samples is not None:
                save_some_examples(
                    generator, 
                    val_dl, 
                    epoch, 
                    folder=args.save_some_samples, 
                    device=device
                )
    

    del generator, discriminator, bpr_net, encoder
    torch.cuda.empty_cache()
    
    generator = GeneratorSkip(conv_filters=[64, 128, 256, 512, 512, 512]).to(device)
    generator.load_state_dict(torch.load(os.path.join(args.weights_dir, f'pix2pixcm_generator_{dataset}.pt'))['model_state_dict'])
    generator.eval()

    encoder = PantsEncoder().to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.weights_dir, f'pix2pixcm_encoder_{dataset}.pt'))['model_state_dict'])
    encoder.eval()

    bpr_net = BPRNet().to(device)
    bpr_net.load_state_dict(torch.load(os.path.join(args.weights_dir, f'pix2pixcm_bpr_{dataset}.pt'))['model_state_dict'])
    bpr_net.eval()

    test_dl = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

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

    for batch in test_loop:
        tshirt = batch['tshirt_image'].to(device)
        true_pant = batch['gt_image'].to(device)

        enc_A, enc_Bg, _ = generator(tshirt)

        enc_B = pants_encoder(true_pant)
        idx = unique_pants.index(batch['pant_id'][0])

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
    config = argparse.ArgumentParser(description="Training and evaluation script for Pix2PixCM.")
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
    config.add_argument('--out_csv', type=str, help="path of the output csv", default=os.path.join(os.getcwd(), 'baselines', 'Pix2PixCM', 'out.csv'))
    config.add_argument('--weights_dir', type=str, help="weights path", default=os.path.join(os.getcwd(), 'baselines', 'Pix2PixCM', 'weights'))

    args = config.parse_args()

    alphas = args.alpha_values if type(args.alpha_values) is list else [args.alpha_values]
    betas = args.beta_values if type(args.beta_values) is list else [args.beta_values]
    mis = args.mi_values if type(args.mi_values) is list else [args.mi_values]
    nis = args.ni_values if type(args.ni_values) is list else [args.ni_values]

    hyperparameter_grid = list(product(alphas, betas, mis, nis))
    hyperparameter_grid = [quadruple for quadruple in hyperparameter_grid]


    if os.path.exists(args.out_csv):
        raise ValueError("Cannot overwrite existing file!")

    with open(args.out_csv, "a") as log_file:
        log_file.write("alpha,beta,mi,ni,mrr10,mrr50,r10,r50,n10,n50\n")
    log_file.close()

    for a, b, m, n in tqdm(hyperparameter_grid, desc='GRID', leave=False):
        with open(args.out_csv, "a") as log_file:
            mrr10, mrr50, r10, r50, n10, n50 = train(a, b, m, n, args)
            log_file.write(f"{a},{b},{m},{n},{mrr10},{mrr50},{r10},{r50},{n10},{n50}\n")
        log_file.close()
        torch.cuda.empty_cache()
        
