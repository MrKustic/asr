import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn


class MaskedBatchNorm(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(n_channels, eps=eps, momentum=momentum)

    def forward(self, input: Tensor, padding_mask: BoolTensor):  # (B, T, D)
        valid_mask = ~padding_mask.squeeze(-1)
        input_flat = input[valid_mask]  # (N, D)

        output_flat = self.batch_norm(input_flat)
        output = torch.zeros_like(input)
        output[valid_mask] = output_flat

        return output  # (B, T, D)


class ConformerConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()

        assert (
            kernel_size > 0 and kernel_size % 2 == 1
        ), f"kernel_size should be the odd positive integer, not {kernel_size}"
        self.layer_norm = nn.LayerNorm(d_model)

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

        self.batch_norm = MaskedBatchNorm(d_model)

        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor, padding_mask: BoolTensor):  # (B, T, D)
        x = self.layer_norm(input)
        x = self.pointwise_conv1(x.transpose(1, 2)).transpose(1, 2)  # (B, T, 2 * D)
        x = F.glu(x, dim=-1)  # (B, T, D)
        x = x.masked_fill(padding_mask, 0.0)

        x = self.depthwise_conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        x = self.batch_norm(x, padding_mask)  # (B, T, D)
        x = F.silu(x)

        x = self.pointwise_conv2(x.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        x = self.dropout(x)

        return x  # (B, T, D)


class FeedForwardConformerModule(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, input: Tensor):
        return self.layers(input)


class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert (
            num_heads > 0 and d_model % num_heads == 0
        ), f"d_model ({d_model}) should be multiple of num_heads ({num_heads})"

        self.feed_forward1 = FeedForwardConformerModule(d_model, ffn_dim, dropout)

        self.convolution = ConformerConvolutionModule(
            d_model=d_model, kernel_size=depthwise_conv_kernel_size, dropout=dropout
        )

        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.multihead_self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)

        self.feed_forward2 = FeedForwardConformerModule(d_model, ffn_dim, dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input: Tensor, padding_mask: BoolTensor):  # (B, T, D)
        x = input + 0.5 * self.feed_forward1(input)

        x = x + self.convolution(x, padding_mask)

        attention_input = self.attention_layer_norm(x)

        attention_output, _ = self.multihead_self_attention(
            query=attention_input,
            key=attention_input,
            value=attention_input,
            key_padding_mask=padding_mask.squeeze(-1),
            need_weights=False,
        )

        x = x + self.attention_dropout(attention_output)

        x = x + 0.5 * self.feed_forward2(x)

        return self.layer_norm(x).masked_fill(padding_mask, 0.0)


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
