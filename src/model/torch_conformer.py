import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.models import Conformer


class TorchConformer(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_tokens: int,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        **kwargs
    ):
        super().__init__()

        self.subsampling = nn.Sequential(
            nn.Conv2d(1, input_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.linear = nn.Linear((n_mels - 3) // 4 * input_dim, input_dim)
        self.conformer = Conformer(
            input_dim,
            num_heads,
            ffn_dim,
            num_layers,
            depthwise_conv_kernel_size,
            **kwargs
        )
        self.head = nn.Linear(input_dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        subsampled = self.subsampling(spectrogram.unsqueeze(1))
        embeds = self.linear(subsampled.flatten(1, 2).transpose(1, 2))
        subsampled_lengths = (spectrogram_length - 3) // 4
        logits, logits_length = self.conformer(embeds, subsampled_lengths)
        log_probs = F.log_softmax(self.head(logits), dim=-1)
        return {"log_probs": log_probs, "log_probs_length": logits_length}
