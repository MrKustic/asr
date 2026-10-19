import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(
        self, min_gain_db: float = -6.0, max_gain_db: float = 6.0, p: float = 0.3
    ):
        super().__init__()
        self._aug = torch_audiomentations.Gain(
            min_gain_in_db=min_gain_db,
            max_gain_in_db=max_gain_db,
            p=p,
            output_type="tensor",
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
