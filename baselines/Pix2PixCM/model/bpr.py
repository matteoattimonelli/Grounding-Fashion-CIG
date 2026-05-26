import torch
import torch.nn as nn
from torch.nn import functional as F


class BPRNet(nn.Module):
    def __init__(
            self,
            in_dim: int = 512,
            out_dim: int = 128,
            metric: str = 'AUC'
    ) -> None:
        super(BPRNet, self).__init__()
        
        self.enc_A_W = nn.Parameter(torch.empty(in_dim, out_dim), requires_grad=True)
        self.enc_A_b = nn.Parameter(torch.empty(out_dim), requires_grad=True)
        self.enc_B_W = nn.Parameter(torch.empty(in_dim, out_dim), requires_grad=True)
        self.enc_B_b = nn.Parameter(torch.empty(out_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.enc_A_W)
        nn.init.xavier_uniform_(self.enc_B_W)
        nn.init.zeros_(self.enc_A_b)
        nn.init.zeros_(self.enc_B_b)

        assert metric in ['AUC', 'MRR']

        self.metric = metric

    def _set_metric(
        self, 
        metric=None
    ) -> None:
        
        assert metric is not None
        self.metric = metric

    def forward(
            self, 
            enc_A_old: torch.Tensor,
            enc_B_old: torch.Tensor,
            enc_B1_old: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        enc_A_avg_pool = F.avg_pool2d(enc_A_old, kernel_size=(2, 2), stride=(1, 1), padding=0).view(-1, 512) # [B, 512]
        enc_A_linear = torch.matmul(enc_A_avg_pool, self.enc_A_W) + self.enc_A_b # [B, 128]
        enc_A = torch.sigmoid(enc_A_linear)

        enc_B_avg_pool = F.avg_pool2d(enc_B_old, kernel_size=(2, 2), stride=(1, 1), padding=0).view(-1, 512) # [B, 512]
        enc_B_linear = torch.matmul(enc_B_avg_pool, self.enc_B_W) + self.enc_B_b # [B, 128]
        enc_B = torch.sigmoid(enc_B_linear) # [B, 128]

        if self.metric == 'AUC':
            enc_B1_avg_pool = F.avg_pool2d(enc_B1_old.view(-1, 512, 2, 2), kernel_size=(2, 2), stride=(1, 1), padding=0).view(enc_B1_old.shape[0], 3, -1)
        else:
            enc_B1_avg_pool = F.avg_pool2d(enc_B1_old, kernel_size=(2, 2), stride=(1, 1), padding=0).view(-1, 512)
        enc_B1_linear = torch.matmul(enc_B1_avg_pool, self.enc_B_W) + self.enc_B_b # [B, 128]
        enc_B1 = torch.sigmoid(enc_B1_linear) # [B, 128]

        return enc_A, enc_B, enc_B1
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = BPRNet().to(device)
    enc = torch.randn(16, 512, 2, 2).to(device)
    enc_B1 = torch.randn(16, 3, 512, 2, 2).to(device)

    enc_A, enc_B, enc_B1 = net(enc, enc, enc_B1)

    ii_sim_ij_v = torch.sum(enc_A*enc_B, dim=1) # [B]
    ii_sim_ik_v = torch.sum(enc_A.unsqueeze(1).repeat(1, 3, 1)*enc_B1, dim=(1, 2))

    print('SUCCESS')