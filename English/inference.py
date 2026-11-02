"""
FastSpeech 2 Inference Script for Synthesizing Speech from Trained Checkpoints.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from scipy.io import wavfile

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize_utterance(model, vocoder, text_seq, speaker_id, configs, control_values=(1.0, 1.0, 1.0)):
    preprocess_config, model_config, _ = configs
    p_control, e_control, d_control = control_values

    text = np.array([text_seq])
    text_lens = np.array([len(text_seq)])
    speakers = np.array([speaker_id])

    batch = [
        ["sample_0"],
        [""],
        speakers,
        text,
        text_lens,
        max(text_lens),
    ]
    batch = to_device(batch, device)

    with torch.no_grad():
        output = model(
            batch[2],
            batch[3],
            batch[4],
            batch[5],
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )
        mel_predicted = output[1].transpose(1, 2)
        lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
        wavs = vocoder_infer(mel_predicted, vocoder, model_config, preprocess_config, lengths=lengths)
        return wavs[0], output[1][0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="FastSpeech 2 Speech Synthesis")
    parser.add_argument("--checkpoint", type=str, default="output/checkpoints/best_model.pth", help="Checkpoint file (.pth / .pth.tar)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed data directory")
    parser.add_argument("--phonemes", type=str, default="{@ð @ɪ @s @sp @ɪ @z @sp @ɐ @sp @t @ɛ @s @t}", help="Phoneme sequence")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker integer ID")
    parser.add_argument("--output_wav", type=str, default="synthesized_output.wav", help="Destination WAV file")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["path"]["preprocessed_path"] = os.path.abspath(args.data_dir)
    elif os.path.exists("/kaggle/working/English_FastSpeech2_Kaggle/preprocessed_data"):
        cfg["path"]["preprocessed_path"] = "/kaggle/working/English_FastSpeech2_Kaggle/preprocessed_data"
    elif not os.path.isabs(cfg["path"]["preprocessed_path"]):
        cfg["path"]["preprocessed_path"] = os.path.join(PACKAGE_ROOT, cfg["path"]["preprocessed_path"])

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        if os.path.exists("/kaggle/working/English/checkpoints/best_model.pth"):
            ckpt_path = "/kaggle/working/English/checkpoints/best_model.pth"
        elif os.path.exists("/kaggle/working/English/checkpoints/latest_checkpoint.pth"):
            ckpt_path = "/kaggle/working/English/checkpoints/latest_checkpoint.pth"

    configs = (
        {"dataset": cfg["dataset"], "path": cfg["path"], "preprocessing": cfg["preprocessing"]},
        cfg["model"],
        cfg["train"],
    )

    cleaners = configs[0]["preprocessing"]["text"]["text_cleaners"]
    seq = text_to_sequence(args.phonemes, cleaners)

    print(f"Loading model checkpoint from {ckpt_path}...")
    model = get_model(args, configs, device, train=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    print("Loading HiFi-GAN vocoder...")
    vocoder = get_vocoder(configs[1], device)

    print("Synthesizing audio...")
    wav, mel = synthesize_utterance(
        model,
        vocoder,
        seq,
        args.speaker_id,
        configs,
        control_values=(args.pitch_control, args.energy_control, args.duration_control),
    )

    sr = configs[0]["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(args.output_wav, sr, wav)
    print(f"Successfully saved synthesized audio to: {args.output_wav} ({len(wav) / sr:.2f} seconds)")


if __name__ == "__main__":
    main()

