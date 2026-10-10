import os
import shutil
from pathlib import Path

import wget

from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


def ensure_librispeech_part(data_dir, part):
    split_dir = data_dir / part
    print(split_dir, split_dir.exists())
    if not split_dir.exists():
        load_librispeech_part(data_dir, part)


def load_librispeech_part(data_dir, part):
    arch_path = data_dir / f"{part}.tar.gz"
    print(f"Loading part {part}")
    wget.download(URL_LINKS[part], str(arch_path))
    shutil.unpack_archive(arch_path, data_dir)
    for fpath in (data_dir / "LibriSpeech").iterdir():
        shutil.move(str(fpath), str(data_dir / fpath.name))
    os.remove(str(arch_path))
    shutil.rmtree(str(data_dir / "LibriSpeech"))


def iterate_librispeech_texts(data_dir=None, parts=None):
    """
    Iterate over the text transcriptions contained in the specified LibriSpeech parts

    Args:
        data_dir (str): path to the LibriSpeech dataset
        parts (list): list of LibriSpeech parts to iterate over
    Yields:
        text (str): next text transcription from the dataset
    """
    assert parts is not None and all(part in URL_LINKS for part in parts)

    if data_dir is None:
        data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        data_dir.mkdir(exist_ok=True, parents=True)
    else:
        data_dir = Path(data_dir)

    for part in parts:
        ensure_librispeech_part(data_dir, part)

    for part in parts:
        for path in (data_dir / part).rglob("*.trans.txt"):
            with path.open() as f:
                for line in f:
                    f_text = " ".join(line.split()[1:]).strip()
                    yield f_text
