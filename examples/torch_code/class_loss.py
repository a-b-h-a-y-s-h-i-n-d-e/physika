import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print

# === Classes ===
class M(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w).float() if not isinstance(w, torch.Tensor) else w.clone().detach().float())

    def forward(self, x):
        x = torch.as_tensor(x).float()
        return (self.w * x)

    def loss(self, y, t):
        return ((y - t) ** 2.0)

# === Program ===