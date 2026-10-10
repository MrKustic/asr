import json
import os
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.librispeech_utils import URL_LINKS, ensure_librispeech_part
from src.utils.io_utils import ROOT_PATH


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, index_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)

        if index_dir is None:
            index_dir = data_dir
        else:
            index_dir = Path(index_dir)
            index_dir.mkdir(exist_ok=True, parents=True)

        self._data_dir = data_dir
        self._index_dir = index_dir

        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train" in part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        ensure_librispeech_part(self._data_dir, part)

        split_dir = self._data_dir / part
        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = sf.info(str(flac_path))
                    length = t_info.frames / t_info.samplerate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
