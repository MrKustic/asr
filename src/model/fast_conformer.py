import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn

from src.model.conformer_blocks import ConformerLayer, FastSubsampling


class FastConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_tokens: int,
        d_model: int,
        n_subsampling_channels: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float,
    ):
        """
        n_mels (int): number of mels in spectrogram.
        n_tokens (int): number of tokens in vocabulary.
        d_model (int): embedding vector lengths.
        n_subsampling_channels (int): number of channels in subsampling block.
        num_heads (int): number of heads in attention layers.
        ffn_dim (int): dimension of feed-forward layers.
        num_layers (int): number of conformer layers.
        depthwise_conv_kernel_size (int): kernel size in depthwise convolution operations.

        """

        super().__init__()

        self.subsampling = FastSubsampling(n_mels, d_model, n_subsampling_channels)

        self.conformer_layers = nn.ModuleList(
            [
                ConformerLayer(
                    d_model, num_heads, ffn_dim, depthwise_conv_kernel_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.head = nn.Linear(d_model, n_tokens)

    def _create_padding_mask(self, x: Tensor, lengths: Tensor):  # (B, T, D)
        padding_mask = torch.arange(x.shape[1], device=x.device) >= lengths.unsqueeze(1)
        return padding_mask.unsqueeze(-1)  # (B, T, 1)

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): padded spectrograms (batch_size, n_mels, n_timeframes)
            spectrogram_length (Tensor): spectrogram original lengths (batch_size,).
        Returns:
            output (dict): output dict containing log_probs and transformed lengths.
        """
        with torch.autocast(
            device_type=spectrogram.device.type,
            enabled=False,
        ):
            embeds = self.subsampling(spectrogram.float())
        subsampled_lengths = (spectrogram_length + 7) // 8  # ceil(length / 8)

        padding_mask = self._create_padding_mask(embeds, subsampled_lengths)
        x = embeds.masked_fill(padding_mask, 0.0)

        for layer in self.conformer_layers:
            x = layer(x, padding_mask)

        logits = self.head(x)
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        return {"log_probs": log_probs, "log_probs_length": subsampled_lengths}
