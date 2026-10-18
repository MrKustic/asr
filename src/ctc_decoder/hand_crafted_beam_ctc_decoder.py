from collections import defaultdict
from dataclasses import dataclass
from heapq import nlargest

import numpy as np
from numpy import ndarray

from src.ctc_decoder.base_ctc_decoder import BaseCTCDecoder


@dataclass
class BeamData:
    """
    A class for storing beam data.

    Attributes:
        prob_blank (float): beam log-probability given it terminates with a blank character.
        prob_non_blank (float): beam log-probability given it terminates with a non-blank character.
        prob_total (float): beam total log-probability
    """

    prob_blank: float = -np.inf
    prob_non_blank: float = -np.inf
    prob_total: float = -np.inf


class BeamSearchCTCDecoder(BaseCTCDecoder):
    """
    Transforms model logits into the prediction texts.
    Implementation of CTC decoder, supports beam search.
    """

    def __init__(self, tokenizer, beam_width=100, token_min_logp=-8.0, **kwargs):
        """
        Args:
            tokenizer (BaseTokenizer): tokenizer model.
            beam_width (int): number of beams.
            token_min_logp (float):
        """
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.token_min_logp = token_min_logp

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
            self._beam_search(emissions[i, :length]) for i, length in enumerate(lengths)
        ]

    def _beam_search(self, log_probs: ndarray):
        """
        Decode one text with beam_search.

        Args:
            log_probs (ndarray): model log_probabilities for entry of shape (n_timeframes, n_tokens).
        Returns:
            text (str): decoded text.
        """
        n_timeframes = log_probs.shape[0]
        beams = defaultdict(BeamData)
        beams[()] = BeamData(
            prob_blank=0.0, prob_non_blank=-np.inf, prob_total=0.0
        )  # single empty beam

        beams.items()

        for t in range(n_timeframes):
            token_ids = np.flatnonzero(log_probs[t] >= self.token_min_logp).tolist()
            token_ids = token_ids if token_ids and token_ids[0] != 0 else token_ids[1:]
            filtered_beams = nlargest(
                self.beam_width, beams.items(), key=lambda b: b[1].prob_total
            )
            new_beams = defaultdict(BeamData)

            for beam_ids, beam in filtered_beams:
                new_beam = new_beams[beam_ids]
                last_token = beam_ids[-1] if beam_ids else -1

                # The decoded text remains unchanged
                prob_blank_delta = beam.prob_total + log_probs[t, 0]  # blank index -- 0
                new_beam.prob_blank = np.logaddexp(
                    new_beam.prob_blank, prob_blank_delta
                )

                if beam_ids:
                    prob_non_blank_delta = (
                        beam.prob_non_blank + log_probs[t, last_token]
                    )
                    new_beam.prob_non_blank = np.logaddexp(
                        new_beam.prob_non_blank, prob_non_blank_delta
                    )

                # The decoded text expands
                prob_non_blank_delta = beam.prob_total + log_probs[t]
                if beam_ids:
                    prob_non_blank_delta[last_token] = (
                        beam.prob_blank + log_probs[t, last_token]
                    )

                for char_idx in token_ids:
                    curr_beam = new_beams[beam_ids + (char_idx,)]
                    curr_beam.prob_non_blank = np.logaddexp(
                        curr_beam.prob_non_blank, prob_non_blank_delta[char_idx]
                    )

            for new_beam in new_beams.values():
                new_beam.prob_total = np.logaddexp(
                    new_beam.prob_blank, new_beam.prob_non_blank
                )

            beams = new_beams

        best_hypothesis_ids = max(beams.items(), key=lambda b: b[1].prob_total)[0]
        return self.tokenizer.ids_to_text(best_hypothesis_ids)
