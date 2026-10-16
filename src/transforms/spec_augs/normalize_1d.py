import torch
from torch import Tensor, nn


class Normalize1d(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True, correction=0)
        normalized = (input - mean) / std.clamp(min=self.eps)

        return torch.where(std < self.eps, 0.0, normalized)
