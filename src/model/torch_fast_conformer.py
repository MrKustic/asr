import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.models import Conformer

from src.model.conformer_blocks import FastSubsampling


class TorchFastConformerEncoder(nn.Module):
    """
    An implementation of the FastConformer model without relative positional attention.
    """

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
        **kwargs
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

        self.conformer = Conformer(
            d_model,
            num_heads,
            ffn_dim,
            num_layers,
            depthwise_conv_kernel_size,
            **kwargs
        )

        self.head = nn.Linear(d_model, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
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

        logits, logits_length = self.conformer(embeds, subsampled_lengths)
        log_probs = F.log_softmax(self.head(logits), dim=-1).float()

        return {"log_probs": log_probs, "log_probs_length": logits_length}
