from collections import defaultdict

import torch
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

    for item in dataset_items:
        result_batch["audio"].append(item["audio"])
        result_batch["text"].append(item["text"])
        result_batch["text_encoded"].append(item["text_encoded"])
        result_batch["text_encoded_length"].append(len(item["text_encoded"]))
        result_batch["audio_path"].append(item["audio_path"])
        result_batch["spectrogram"].append(item["spectrogram"])
        result_batch["spectrogram_length"].append(item["spectrogram"].shape[1])

    result_batch["audio"] = torch.cat(result_batch["audio"])
    result_batch["text_encoded"] = pad_sequence(
        result_batch["text_encoded"], batch_first=True
    )
    result_batch["text_encoded_length"] = torch.tensor(
        result_batch["text_encoded_length"], dtype=torch.int32
    )
    result_batch["spectrogram"] = torch.cat(result_batch["spectrogram"])
    result_batch["spectrogram_length"] = torch.tensor(
        result_batch["spectrogram_length"], dtype=torch.int32
    )

    return result_batch
