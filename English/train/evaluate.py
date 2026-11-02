import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    last_batch = None
    last_output = None
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
            last_batch = batch
            last_output = output

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = (
        f"Validation Step {step}, Total Loss: {loss_means[0]:.4f}, "
        f"Mel Loss: {loss_means[1]:.4f}, Mel PostNet Loss: {loss_means[2]:.4f}, "
        f"Pitch Loss: {loss_means[3]:.4f}, Energy Loss: {loss_means[4]:.4f}, "
        f"Duration Loss: {loss_means[5]:.4f}"
    )

    if logger is not None and vocoder is not None and last_batch is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            last_batch,
            last_output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag=f"Validation/step_{step}_{tag}",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag=f"Validation/step_{step}_{tag}_reconstructed",
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag=f"Validation/step_{step}_{tag}_synthesized",
        )
        try:
            import matplotlib.pyplot as plt
            if fig is not None:
                plt.close(fig)
            plt.close('all')
        except Exception:
            pass

    if logger is not None:
        try:
            logger.flush()
        except Exception:
            pass
    del last_batch, last_output, loader, dataset, Loss
    import gc
    gc.collect()
    return message, loss_means


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    configs = (
        {"dataset": cfg["dataset"], "path": cfg["path"], "preprocessing": cfg["preprocessing"]},
        cfg["model"],
        cfg["train"],
    )

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(configs[1], device)
    message, _ = evaluate(model, args.restore_step, configs, vocoder=vocoder)
    print(message)

