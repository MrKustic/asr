import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ctc_loss = CTCLoss(**kwargs)

    def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch) -> dict:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = self.ctc_loss.forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}
