from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = defaultdict(list)
    result_batch["spectrogram_length"] = torch.tensor(
        [item["spectrogram"].shape[-1] for item in dataset_items], dtype=torch.int32
    )
    max_length = result_batch["spectrogram_length"].max()

    for item in dataset_items:
        result_batch["audio"].append(item["audio"])
        result_batch["text"].append(item["text"])
        result_batch["text_encoded"].append(item["text_encoded"])
        result_batch["text_encoded_length"].append(len(item["text_encoded"]))
        result_batch["audio_path"].append(item["audio_path"])

        spec = item["spectrogram"]
        padded_spec = F.pad(spec, (0, max_length - spec.shape[-1]), value=spec.min())
        result_batch["spectrogram"].append(padded_spec)

    result_batch["text_encoded"] = pad_sequence(
        result_batch["text_encoded"], batch_first=True
    )
    result_batch["text_encoded_length"] = torch.tensor(
        result_batch["text_encoded_length"], dtype=torch.int32
    )
    result_batch["spectrogram"] = torch.cat(result_batch["spectrogram"])

    return result_batch
