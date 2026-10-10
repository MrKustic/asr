from pathlib import Path

from hydra.utils import instantiate

from src.text_encoder.utils import normalize_text
from src.utils.io_utils import ROOT_PATH


def train_tokenizer(config, model_path):
    text_iterator = instantiate(config.text_iterator)
    normalized_text_iterator = (normalize_text(text) for text in text_iterator)

    model_path.parent.mkdir(exist_ok=True, parents=True)
    instantiate(
        config.tokenizer_trainer,
        sentence_iterator=normalized_text_iterator,
        model_prefix=model_path.with_suffix(""),
    )


def ensure_tokenizer(config):
    model_path = config.get("model_path", None)
    if model_path is not None:
        model_path = Path(model_path)
    else:
        model_path = ROOT_PATH / "data" / "tokenizers" / "tokenizer.model"

    mode = config.get("mode", "train_if_missing")
    assert mode in ["train_if_missing", "train", "none"]

    if model_path.exists() and mode == "train_if_missing":
        return model_path
    train_tokenizer(config, model_path)
    return model_path


def build_text_encoder(config):
    if config.tokenizer.mode == "none":
        return instantiate(config.text_encoder)

    model_path = ensure_tokenizer(config.tokenizer)
    return instantiate(config.text_encoder, model_path=str(model_path))
