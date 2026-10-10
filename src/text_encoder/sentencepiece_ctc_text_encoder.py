from itertools import groupby

import torch
from sentencepiece import SentencePieceProcessor

from src.text_encoder.utils import normalize_text


class SentencePieceCTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, model_path, **kwargs):
        """
        Args:
            model_path (str): path to SentencePiece tokenizer model
        """

        self.sp = SentencePieceProcessor(model_path)
        self.vocab = [self.EMPTY_TOK] + [
            self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())
        ]

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.vocab[item]

    def encode(self, text) -> torch.Tensor:
        text = normalize_text(text)
        return torch.tensor(self.sp.encode_as_ids(text), dtype=torch.int32) + 1

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        sp_inds = [ind - 1 for ind in inds if ind != 0]
        return self.sp.decode_ids(sp_inds)

    def ctc_decode(self, inds) -> str:
        """
        Decode text using CTC.

        Args:
            inds (list): list of tokens.
        Returns:
            text (str): decoded text without empty tokens or repetitions.
        """
        non_repeating_sp_inds = [key - 1 for key, _ in groupby(inds) if key != 0]
        return self.sp.decode_ids(non_repeating_sp_inds)
