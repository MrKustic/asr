from src.ctc_decoder.base_ctc_decoder import BaseCTCDecoder
from src.tokenizer.base_tokenizer import BaseTokenizer


class ArgmaxCTCDecoder(BaseCTCDecoder):
    """
    Transforms model logits into the prediction texts.

    Chooses the argmax token for each timeframe.
    """

    def __init__(self, tokenizer: BaseTokenizer, **kwargs):
        """
        Args:
            tokenizer (BaseTokenizer): toenizer class
        """
        self.tokenizer = tokenizer

    def decode(self, log_probs, log_probs_length, **batch):
        """
        Decodes the log probabilities into text.

        Args:
            log_probs (Tensor): padded model log probabilities of shape (batch_size, n_timeframes, n_tokens).
            log_probs_length (Tensor): original lengths of log_probs of shape (batch_size,).
        Returns:
            texts (List[str]): a list of strings containing prediction texts for each sample in the batch.
        """
        raw_preds = log_probs.argmax(dim=-1).detach().cpu()
        lengths = log_probs_length.detach().cpu()
        return [
            self._ctc_decode_sample(ids[:ids_length])
            for ids, ids_length in zip(raw_preds, lengths)
        ]

    def _ctc_decode_sample(self, ids):
        """
        Decode text using CTC.

        Args:
            ids (Tensor): sequence of token indices to CTC decode.
        Returns:
            text (str): decoded text without empty tokens or repetitions.
        """
        indices_mask = ids != 0
        indices_mask[1:] &= ids[1:] != ids[:-1]
        return self.tokenizer.ids_to_text(ids[indices_mask])
