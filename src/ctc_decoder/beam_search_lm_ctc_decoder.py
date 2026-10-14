from pyctcdecode.decoder import build_ctcdecoder

from src.ctc_decoder.base_ctc_decoder import BaseCTCDecoder
from src.tokenizer.base_tokenizer import BaseTokenizer


class BeamSearchLMCTCDecoder(BaseCTCDecoder):
    """
    Transforms model logits into the prediction texts.
    Uses pyctcdecode CTCDecoder, supports LM model and beam search.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        kenlm_model_path=None,
        alpha=0.5,
        beta=1.0,
        beam_width=100,
        **kwargs
    ):
        """
        Args:
            tokenizer (BaseTokenizer): tokenizer model.
            kenlm_model_path (str | None): if not None, path to pretrained KenLM model.
            alpha (float): weight for language model during shallow fusion.
            beta (float): weight for length score adjustment of during scoring.
        """
        self.decoder = build_ctcdecoder(
            labels=tokenizer.vocab,
            kenlm_model_path=kenlm_model_path,
            alpha=alpha,
            beta=beta,
        )
        self.beam_width = beam_width

    def decode(self, log_probs, log_probs_length, **batch):
        """
        Decodes the log probabilities into text.

        Args:
            log_probs (Tensor): padded model log probabilities of shape (batch_size, n_timeframes, n_tokens).
            log_probs_length (Tensor): original lengths of log_probs of shape (batch_size,).
        Returns:
            texts (List[str]): a list of strings containing prediction texts for each sample in the batch.
        """
        emissions = log_probs.detach().float().cpu().numpy()
        lengths = log_probs_length.detach().cpu().tolist()
        return [
            self.decoder.decode(emissions[i, :length], beam_width=self.beam_width)
            for i, length in enumerate(lengths)
        ]
