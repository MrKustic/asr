import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.tokenizer.prepare_tokenizer import build_tokenizer
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    """
    Main script for calculating metircs. Instantiates metrics and
    dataloaders. Runs Inferencer to calculate metrics.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    device = "cpu"

    # setup tokenizer and ctc_decoder
    tokenizer = build_tokenizer(config)
    ctc_decoder = instantiate(config.ctc_decoder, tokenizer=tokenizer)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, tokenizer, device)

    # model is not used to compute metrics.
    model = None

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        metrics["inference"].append(instantiate(metric_config))

    # path to load model predictions from
    if config.inferencer.save_path is None:
        save_path = ROOT_PATH / "data" / "predictions"
    else:
        save_path = Path(config.inferencer.save_path)
    assert save_path.exists(), f"dir with predictions '{save_path}' should exists."

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        tokenizer=tokenizer,
        ctc_decoder=ctc_decoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=True,
    )

    logs = inferencer.run_inference(calc_metrics_only=True)

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
