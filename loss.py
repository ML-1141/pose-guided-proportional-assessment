import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineEmbeddingLossMargin(nn.Module):
    """
    Loss(X, Y) = { 1 - cos(X1, X2),          if Y =  1
                   max(0, cos(X1, X2)-m),    if Y = -1 }
    """
    def __init__(self, margin: float = 0.7):
        super().__init__()
        self.margin = float(margin)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1:
            y = y.view(-1)
        y = y.to(x1.device)

        cos_sim = F.cosine_similarity(x1, x2, dim=1)
        loss_pos = 1.0 - cos_sim
        loss_neg = torch.clamp(cos_sim - self.margin, min=0.0)
        loss = torch.where(y == 1, loss_pos, loss_neg)
        return loss.mean()
