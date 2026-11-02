"""
Comparative Evaluation and Audio Synthesis Script for English FastSpeech 2:
Compares Best Validation Checkpoint (approx. Step 16,000) vs Latest Checkpoint (Step 23,000)
on Identical Validation Examples and Synthesizes Benchmark Audio with HiFi-GAN.
"""

import os
import sys
import argparse
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device
from model import FastSpeech2Loss
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_checkpoint_on_samples(model, Loss, samples, configs):
    preprocess_config, model_config, _ = configs
    model.eval()
    losses_accum = [0.0] * 6
    generated_mels = []

    with torch.no_grad():
        for batch in samples:
            batch = to_device(batch, device)
            output = model(*(batch[2:]))
            losses = Loss(batch, output)
            for i in range(len(losses)):
                losses_accum[i] += losses[i].item()
            
            # Postnet mel output
            postnet_mel = output[1].transpose(1, 2)
            mel_len = output[9]
            generated_mels.append((batch[0], postnet_mel, mel_len))

    n = len(samples)
    mean_losses = [l / n for l in losses_accum]
    return mean_losses, generated_mels


def run_comparison(best_ckpt_path, latest_ckpt_path, config_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocess_config = {"dataset": cfg["dataset"], "path": cfg["path"], "preprocessing": cfg["preprocessing"]}
    model_config = cfg["model"]
    train_config = cfg["train"]
    configs = (preprocess_config, model_config, train_config)

    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    vocoder = get_vocoder(model_config, device)

    # Load validation dataset
    val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    
    # Select 10 fixed diverse validation samples across different speakers
    fixed_indices = [0, 25, 50, 100, 150, 200, 300, 400, 500, 600]
    fixed_indices = [idx for idx in fixed_indices if idx < len(val_dataset)]
    fixed_samples = [[val_dataset[idx]] for idx in fixed_indices]

    results = {}

    for label, ckpt_p in [("Best_Step_16000", best_ckpt_path), ("Latest_Step_23000", latest_ckpt_path)]:
        if not os.path.isfile(ckpt_p):
            print(f"[Notice] Checkpoint {ckpt_p} not found on disk. Skipping physical weight evaluation.")
            continue

        print(f"\nEvaluating {label} ({ckpt_p})...")
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        step_num = ckpt.get("step", "Unknown")
        val_l = ckpt.get("val_loss", ckpt.get("best_loss", "Unknown"))

        # Build fresh model
        model, _ = get_model(argparse.Namespace(restore_step=0), configs, device, train=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model"])
        raw_model.eval()

        mean_losses, generated_mels = evaluate_checkpoint_on_samples(model, Loss, fixed_samples, configs)
        results[label] = {
            "step": step_num,
            "recorded_val_loss": val_l,
            "sample_eval_losses": {
                "total": mean_losses[0],
                "mel": mean_losses[1],
                "postnet": mean_losses[2],
                "pitch": mean_losses[3],
                "energy": mean_losses[4],
                "duration": mean_losses[5],
            }
        }

        # Synthesize audio for first 3 samples
        sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        for sample_idx in range(min(3, len(generated_mels))):
            base_id = generated_mels[sample_idx][0][0]
            mel = generated_mels[sample_idx][1]
            mel_len = generated_mels[sample_idx][2]
            lengths = mel_len * preprocess_config["preprocessing"]["stft"]["hop_length"]
            wav = vocoder_infer(mel, vocoder, model_config, preprocess_config, lengths=lengths)[0]
            wav_out_path = os.path.join(output_dir, f"{label}_sample_{sample_idx}_{base_id}.wav")
            wavfile.write(wav_out_path, sr, wav)
            print(f"  -> Generated: {wav_out_path}")

    # Output comparison report
    report_path = os.path.join(output_dir, "checkpoint_comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison report saved to: {report_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_ckpt", type=str, default="/kaggle/working/English/checkpoints/best_model.pth")
    parser.add_argument("--latest_ckpt", type=str, default="/kaggle/working/English/checkpoints/latest_checkpoint.pth")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoint_comparison_results")
    args = parser.parse_args()
    run_comparison(args.best_ckpt, args.latest_ckpt, args.config, args.output_dir)
