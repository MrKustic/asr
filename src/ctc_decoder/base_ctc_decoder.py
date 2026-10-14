from abc import ABC, abstractmethod


class BaseCTCDecoder(ABC):
    """
    Base class for all CTC decoders.
    Transforms model logits into the prediction texts.
    """

    @abstractmethod
    def decode(self, log_probs, log_probs_length, **batch):
        """
        Decodes the log probabilities into text.

        Args:
            log_probs (Tensor): padded model log probabilities of shape (batch_size, n_timeframes, n_tokens).
            log_probs_length (Tensor): original lengths of log_probs of shape (batch_size,).
        Returns:
            texts (List[str]): a list of strings containing prediction texts for each sample in the batch.
        """
        raise NotImplementedError()
