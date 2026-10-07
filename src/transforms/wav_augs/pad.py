import torch
from torch import Tensor, nn


class Pad(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, data: Tensor):
        length_delta = self.length - data.shape[-1]
        return nn.functional.pad(data, (0, length_delta))
