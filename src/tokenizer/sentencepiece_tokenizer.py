import torch
from sentencepiece import SentencePieceProcessor

from src.tokenizer.base_tokenizer import BaseTokenizer
from src.utils.text import normalize_text


class SentencePieceTokenizer(BaseTokenizer):
    """
    A tokenizer that uses SentencePiece.
    """

    EMPTY_TOK = ""

    def __init__(self, model_path, **kwargs):
        """
        Args:
            model_path (str): path to SentencePiece tokenizer model
        """

        self.sp = SentencePieceProcessor(model_path)
        self._vocab = [self.EMPTY_TOK] + [
            self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())
        ]

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
        return torch.tensor(self.sp.encode_as_ids(text), dtype=torch.int32) + 1

    def ids_to_text(self, ids):
        """
        Decodes token ids back into text.

        Args:
            ids (list[int] | Tensor): sequence of token ids to decode.
        Returns:
            text (str): text decoded from the ids.
        """
        sp_inds = [int(ind) - 1 for ind in ids if int(ind) != 0]
        return self.sp.decode_ids(sp_inds)
