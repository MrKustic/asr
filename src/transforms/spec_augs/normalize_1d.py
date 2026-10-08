import torch
from torch import Tensor, nn


class Normalize1d(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True, correction=0)
        return (input - mean) / (std + torch.tensor(self.eps, dtype=torch.float32))
