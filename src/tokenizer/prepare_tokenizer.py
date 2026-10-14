from pathlib import Path

from hydra.utils import instantiate

from src.utils.io_utils import ROOT_PATH
from src.utils.text import normalize_text


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
    mode = config.get("mode", "train_if_missing")
    assert mode in [
        "train_if_missing",
        "train",
    ], f"Supported modes for ensure_tokenizer are 'train' and 'train_if_missing', not '{mode}'"

    model_path = config.get("model_path", None)
    if model_path is not None:
        model_path = Path(model_path)
    else:
        model_path = ROOT_PATH / "data" / "tokenizers" / "tokenizer.model"

    if model_path.exists() and mode == "train_if_missing":
        return model_path
    train_tokenizer(config, model_path)
    return model_path


def build_tokenizer(config):
    if config.tokenizer_training.mode == "none":
        return instantiate(config.tokenizer)

    model_path = ensure_tokenizer(config.tokenizer_training)
    return instantiate(config.tokenizer, model_path=str(model_path))
