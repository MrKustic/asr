from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """
    Base class for all tokenizers.
    """

    @property
    @abstractmethod
    def vocab(self) -> list[str]:
        """
        Returns the vocabulary.

        Returns:
            vocab (list[str]): list of words in the vocabulary. The blank token should have index 0.
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.vocab[item]

    @abstractmethod
    def text_to_ids(self, text):
        """
        Encode text into a sequence of tokens.

        Args:
            text (str): text to encode.
        Returns:
            tokens (Tensor): sequence of tokens ids representing the provided text.
        """
        raise NotImplementedError()

    @abstractmethod
    def ids_to_text(self, ids):
        """
        Decodes token ids back into text.

        Args:
            ids (list[int] | Tensor): sequence of token ids to decode.
        Returns:
            text (str): text decoded from the ids.
        """
        raise NotImplementedError()
