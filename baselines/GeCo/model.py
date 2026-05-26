import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
import types


def _forward_impl(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    features = torch.flatten(x, 1)
    return features


class GeCo(nn.Module):
    def __init__(
            self,
            emb_dim: int = 64,
            learning_rate: float = 1e-6,
            alpha: float = 1,
            beta: float = 1,
            gamma: float = 0.01,
            temperature: float = 0.07
    ):
        super(GeCo, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.forward = types.MethodType(_forward_impl, self.resnet)

        self.query_proj = nn.Linear(self.resnet.fc.in_features * 2, emb_dim)
        self.bottom_proj = nn.Linear(self.resnet.fc.in_features, emb_dim)

        self.query_proj.apply(self.init_weights)
        self.bottom_proj.apply(self.init_weights)

        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.99)
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def init_weights(
            self, 
            layer
        ) -> None:
        
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward_bottom(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        
        x = self.resnet(x)
        return self.bottom_proj(x)

    def forward_query(
            self,
            top: torch.Tensor,
            template: torch.Tensor
    ) -> torch.Tensor:
        top_emb = self.resnet(top)
        temp_emb = self.resnet(template)
        whole_emb = torch.hstack([top_emb, temp_emb])
        return self.query_proj(whole_emb)

    def forward(
            self,
            top: torch.Tensor,
            pos: torch.Tensor,
            neg: torch.Tensor,
            template: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        query_emb = self.forward_query(top, template)
        pos_emb = self.forward_bottom(pos)
        neg_emb = self.forward_bottom(neg)
        return query_emb, pos_emb, neg_emb

    def l2_reg_loss(
            self,
            *args
    ) -> torch.Tensor:
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2) / emb.shape[0]
        for param in self.parameters():
            emb_loss += torch.norm(param, p=2) / param.shape[0]
        return emb_loss * self.gamma

    def InfoNCE(
            self,
            query_emb: torch.Tensor, 
            cand_emb: torch.Tensor, 
            b_cos: bool = True
    ) -> torch.Tensor:

        if b_cos:
            query_emb, cand_emb = torch.nn.functional.normalize(query_emb, dim=1), torch.nn.functional.normalize(cand_emb, dim=1)

        cand_score = (query_emb @ cand_emb.T) / self.temperature
        score = torch.diag(torch.nn.functional.log_softmax(cand_score, dim=1))
        return -score.mean()

    @staticmethod
    def bpr_loss(
        query_emb: torch.Tensor, 
        pos_item_emb: torch.Tensor, 
        neg_item_emb: torch.Tensor
    ) -> torch.Tensor:
        pos_score = torch.mul(query_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(query_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss), pos_score, neg_score

    def get_cost_updates(
            self,
            top_img: torch.Tensor,
            pos_img: torch.Tensor,
            neg_img: torch.Tensor,
            template: torch.Tensor,
            backprop: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        query_emb, pos_emb, neg_emb = self(top_img, pos_img, neg_img, template)

        bpr_loss, pos_scores, neg_scores = self.bpr_loss(query_emb, pos_emb, neg_emb)
        cl_loss = self.InfoNCE(query_emb, pos_emb, 0.07)
        reg_loss = self.l2_reg_loss(query_emb, pos_emb, neg_emb)

        cost = self.alpha * bpr_loss + self.beta * cl_loss + reg_loss

        if backprop:
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

        return cost, pos_scores, neg_scores


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GeCo().to(device)
    img = torch.randn(16, 3, 128, 128).to(device)

    cost, pos_scores, neg_scores = model.get_cost_updates(img, img, img, img)
    print('SUCCESS!')
