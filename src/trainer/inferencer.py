from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.ctc_decoder.base_ctc_decoder import BaseCTCDecoder
from src.metrics.tracker import MetricTracker
from src.tokenizer.base_tokenizer import BaseTokenizer
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        tokenizer,
        ctc_decoder,
        save_path,
        metrics,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            tokenizer (BaseTokenizer): tokenizer.
            ctc_decoder (BaseCTCDecoder): ctc_decoder.
            save_path (Path): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.tokenizer: BaseTokenizer = tokenizer
        self.ctc_decoder: BaseCTCDecoder = ctc_decoder

        # define amp
        self.device_type = torch.device(self.device).type
        amp_config = self.cfg_trainer.get("amp", {})
        self.amp_enabled = amp_config.get("enabled", False)

        dtype_name = amp_config.get("dtype", "float16")
        str2dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        assert (
            dtype_name in str2dtype
        ), "Only 'float16' and 'bfloat16' are supported for the AMP data type."
        self.amp_dtype = str2dtype[dtype_name]

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics["inference"]],
            writer=None,
        )

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        with torch.autocast(self.device_type, self.amp_dtype, enabled=self.amp_enabled):
            outputs = self.model(**batch)
        batch.update(outputs)
        batch.update({"prediction_text": self.ctc_decoder.decode(**batch)})

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            batch_size = batch["log_probs"].shape[0]
            for i in range(batch_size):
                prediction_text = batch["prediction_text"][i]
                audio_path = batch["audio_path"][i]
                file_name = f"{Path(audio_path).stem}.txt"

                with (self.save_path / part / file_name).open("w") as f:
                    print(prediction_text, file=f)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
