import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.conformer_blocks import ConformerLayer


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_tokens: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.subsampling = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.linear = nn.Linear((n_mels - 3) // 4 * d_model, d_model)

        self.conformer_layers = nn.ModuleList(
            [
                ConformerLayer(
                    d_model, num_heads, ffn_dim, depthwise_conv_kernel_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.head = nn.Linear(d_model, n_tokens)

    def _create_padding_mask(self, x: Tensor, lengths: Tensor):
        padding_mask = torch.arange(x.shape[1], device=x.device) >= lengths.unsqueeze(1)
        return padding_mask.unsqueeze(-1)

    def forward(self, spectrogram, spectrogram_length, **batch):
        subsampled = self.subsampling(spectrogram.unsqueeze(1))
        embeds = self.linear(subsampled.flatten(1, 2).transpose(1, 2))
        subsampled_lengths = (spectrogram_length - 3) // 4

        padding_mask = self._create_padding_mask(embeds, subsampled_lengths)
        x = embeds.masked_fill(padding_mask, 0.0)

        for layer in self.conformer_layers:
            x = layer(x, padding_mask)

        logits = self.head(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": subsampled_lengths}
