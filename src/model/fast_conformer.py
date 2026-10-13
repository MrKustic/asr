import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.models import Conformer


class FastSubsampling(nn.Module):
    """
    FastConformer Subsampling Layer.
    """

    def __init__(self, input_dim: int, output_dim: int, n_channels: int):
        """
        Args:
            input_dim (int): number of input features.
            output_dim (int): number of output features.
            n_channels (int): numver of channels in convolution layers.
        """
        super().__init__()

        layers = [
            nn.Conv2d(1, n_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        ]

        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(
                        n_channels,
                        n_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=n_channels,
                    ),
                    nn.Conv2d(n_channels, n_channels, kernel_size=1, stride=1),
                    nn.ReLU(),
                ]
            )

        self.conv = nn.Sequential(*layers)

        n_conv_mels = (input_dim + 7) // 8  # ceil(input_dim / 8)
        self.linear = nn.Linear(n_conv_mels * n_channels, output_dim)

    def forward(self, input: Tensor):
        """
        Class forward method.
        Args:
            input (Tensor): tensor of shape (batch_size, input_dim, n_tokens).
        Returns:
            embedding (Tensor): subsampled input tensor of shape (batch_size, n_tokens', output_dim).
        """
        feat_map = self.conv(input.unsqueeze(1))
        return self.linear(feat_map.flatten(1, 2).transpose(1, 2))


class FastConformerEncoder(nn.Module):
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
        embeds = self.subsampling(spectrogram)
        subsampled_lengths = (spectrogram_length + 7) // 8  # ceil(length / 8)

        logits, logits_length = self.conformer(embeds, subsampled_lengths)
        log_probs = F.log_softmax(self.head(logits), dim=-1).float()

        return {"log_probs": log_probs, "log_probs_length": logits_length}
