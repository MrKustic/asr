from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        assert (
            transcription_dir is None or Path(transcription_dir).exists()
        ), f"Provided transcription_dir '{transcription_dir}' does not exist."

        data = []
        for path in Path(audio_dir).iterdir():
            if path.suffix not in [".mp3", ".wav", ".flac"]:
                continue
            entry: dict = {"text": "", "path": str(path)}

            t_info = sf.info(str(path))
            length = t_info.frames / t_info.samplerate
            entry["audio_len"] = length

            if transcription_dir:
                transc_path = Path(transcription_dir) / (path.stem + ".txt")
                assert (
                    transc_path.exists()
                ), f"Transcription for '{path}' does not exist."
                with transc_path.open() as f:
                    entry["text"] = f.read().strip()

            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
