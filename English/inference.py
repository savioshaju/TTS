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
INFERENCE_PKG = os.path.join(PACKAGE_ROOT, "inference")
BASE_DIR = os.path.dirname(PACKAGE_ROOT)
for p in [INFERENCE_PKG, PACKAGE_ROOT, BASE_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint file (.pth / .pth.tar)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed data directory")
    parser.add_argument("--text", type=str, default=None, help="Raw English text to synthesize")
    parser.add_argument("--phonemes", type=str, default=None, help="Phoneme sequence, e.g. '{@ð @ɪ @s @sp @ɪ @z @sp @ɐ @sp @t @ɛ @s @t}'")
    parser.add_argument("--speaker_id", type=int, default=65, help="Speaker integer ID (default: 65 = spk_067)")
    parser.add_argument("--output_wav", type=str, default="synthesized_output.wav", help="Destination WAV file")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.378, help="Duration control (default: 1.378 for calibrated pacing)")
    parser.add_argument("--test_all", action="store_true", help="Synthesize all official assignment benchmark test sentences")
    args = parser.parse_args()

    if args.test_all:
        import generate_assignment_audios
        print("Running full official English benchmark synthesis...")
        generate_assignment_audios.main()
        return

    config_path = args.config
    if not os.path.exists(config_path):
        candidate = os.path.join(PACKAGE_ROOT, "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            candidate2 = os.path.join(PACKAGE_ROOT, "English", "config.yaml")
            if os.path.exists(candidate2):
                config_path = candidate2

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["path"]["preprocessed_path"] = os.path.abspath(args.data_dir)
    elif os.path.exists("/kaggle/working/English_FastSpeech2_Kaggle/preprocessed_data"):
        cfg["path"]["preprocessed_path"] = "/kaggle/working/English_FastSpeech2_Kaggle/preprocessed_data"
    elif not os.path.isabs(cfg["path"]["preprocessed_path"]):
        cfg["path"]["preprocessed_path"] = os.path.join(PACKAGE_ROOT, cfg["path"]["preprocessed_path"])

    # Fallback to pre-bundled stats.json and speakers.json in repository if preprocessed_data is absent
    prep_path = cfg["path"]["preprocessed_path"]
    if not os.path.exists(os.path.join(prep_path, "stats.json")):
        for candidate in [PACKAGE_ROOT, os.path.join(PACKAGE_ROOT, "inference")]:
            if os.path.exists(os.path.join(candidate, "stats.json")):
                cfg["path"]["preprocessed_path"] = candidate
                break

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        candidates = [
            os.path.join(PACKAGE_ROOT, "checkpoints", "best_model.pth"),
            os.path.join(PACKAGE_ROOT, "..", "checkpoints", "best_model.pth"),
            os.path.join(PACKAGE_ROOT, "output", "checkpoints", "best_model.pth"),
            "/kaggle/working/English/checkpoints/best_model.pth",
            "/kaggle/working/English/checkpoints/latest_checkpoint.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {args.checkpoint} or common locations.")
        print("Please download best_model.pth from Google Drive to English/checkpoints/best_model.pth")
        sys.exit(1)

    phoneme_input = args.phonemes
    if args.text is not None:
        try:
            import synthesize_custom
            lexicon = synthesize_custom.load_mfa_lexicon()
            phoneme_input = synthesize_custom.text_to_phonemes(args.text, lexicon)
            print(f"Synthesizing English text: \"{args.text}\"")
            print(f"Generated phoneme sequence: {phoneme_input}")
        except Exception as e:
            print(f"Warning: Text-to-phoneme conversion fallback: {e}")
            phoneme_input = args.phonemes

    if not phoneme_input:
        phoneme_input = "{@ð @ɪ @s @sp @ɪ @z @sp @ɐ @sp @t @ɛ @s @t}"

    configs = (
        {"dataset": cfg["dataset"], "path": cfg["path"], "preprocessing": cfg["preprocessing"]},
        cfg["model"],
        cfg["train"],
    )

    cleaners = configs[0]["preprocessing"]["text"]["text_cleaners"]
    seq = text_to_sequence(phoneme_input, cleaners)

    print(f"Loading model checkpoint from {ckpt_path}...")
    model = get_model(args, configs, device, train=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

