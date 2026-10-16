import torchaudio
from torch import Tensor, nn


class SpeedPertrubation(nn.Module):
    def __init__(self, orig_freq: int, factors: list):
        super().__init__()
        self._aug = torchaudio.transforms.SpeedPerturbation(orig_freq, factors)

    def __call__(self, input: Tensor):
        return self._aug(input)[0]
