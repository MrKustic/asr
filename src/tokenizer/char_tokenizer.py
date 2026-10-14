from string import ascii_lowercase

import torch

from src.tokenizer.base_tokenizer import BaseTokenizer
from src.utils.text import normalize_text


class CharTokenizer(BaseTokenizer):
    """
    A default char tokenizer.
    """

    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self._vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    @property
    def vocab(self):
        """
        Returns the vocabulary.

        Returns:
            vocab (list[str]): list of words in the vocabulary. The blank token should have index 0.
        """
        return self._vocab

    def text_to_ids(self, text):
        """
        Normalise text and encode it into a sequence of tokens.

        Args:
            text (str): text to encode.
        Returns:
            tokens (Tensor): sequence of tokens ids representing the provided text.
        """
        text = normalize_text(text)
        try:
            return torch.tensor(
                [self.char2ind[char] for char in text], dtype=torch.int32
            )
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def ids_to_text(self, ids):
        """
        Decodes token ids back into text.

        Args:
            ids (list[int] | Tensor): sequence of token ids to decode.
        Returns:
            text (str): text decoded from the ids.
        """
        return "".join([self.ind2char[int(ind)] for ind in ids]).strip()
