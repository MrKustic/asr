from typing import List

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from src.utils.text import normalize_text


class CERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, prediction_text: List[str], text: List[str], **kwargs):
        cer_sum = 0.0
        for pred_text, target_text in zip(prediction_text, text):
            target_text = normalize_text(target_text)
            pred_text = normalize_text(pred_text)
            cer_sum += calc_cer(target_text, pred_text)
        return cer_sum / len(text)
