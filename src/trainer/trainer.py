from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer
from src.utils.text import normalize_text


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with torch.autocast(self.device_type, self.amp_dtype, enabled=self.amp_enabled):
            outputs = self.model(**batch)
        batch.update(outputs)

        if not self.is_train or metric_funcs:
            batch.update({"prediction_text": self.ctc_decoder.decode(**batch)})

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            self.scaler.scale(
                batch["loss"]
            ).backward()  # sum of all losses is always called loss

            self.scaler.unscale_(self.optimizer)
            self._clip_grad_norm()

            scale_before = self.scaler.get_scale()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            did_optimizer_step = self.scaler.get_scale() >= scale_before
            if self.lr_scheduler is not None and did_optimizer_step:
                self.lr_scheduler.step()
        else:
            did_optimizer_step = None

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch, did_optimizer_step

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_audio(self, audio, audio_path, **batch):
        audio_path = audio_path[0]
        sample_rate = sf.info(audio_path).samplerate
        self.writer.add_audio(Path(audio_path).name, audio[0], sample_rate=sample_rate)

    def log_predictions(
        self, text, prediction_text, audio_path, examples_to_log=10, **batch
    ):
        tuples = list(zip(prediction_text, text, audio_path))

        rows = {}
        for pred, target, audio_path in tuples[:examples_to_log]:
            target = normalize_text(target)
            pred = normalize_text(pred)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "step": self.writer.step,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
