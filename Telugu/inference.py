"""
FastSpeech 2 Inference Script for Telugu Speech Synthesis.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import unicodedata
import re
from scipy.io import wavfile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

TELUGU_VOWELS = {
    '\u0c05': 'a', '\u0c06': 'aa', '\u0c07': 'i', '\u0c08': 'ii',
    '\u0c09': 'u', '\u0c0a': 'uu', '\u0c0b': 'ru', '\u0c0e': 'e',
    '\u0c0f': 'ee', '\u0c10': 'ai', '\u0c12': 'o', '\u0c13': 'oo', '\u0c14': 'au'
}
TELUGU_MATRAS = {
    '\u0c3e': 'aa', '\u0c3f': 'i', '\u0c40': 'ii', '\u0c41': 'u',
    '\u0c42': 'uu', '\u0c43': 'ru', '\u0c46': 'e', '\u0c47': 'ee',
    '\u0c48': 'ai', '\u0c4a': 'o', '\u0c4b': 'oo', '\u0c4c': 'au'
}
TELUGU_CONSONANTS = {
    '\u0c15': 'k',   '\u0c16': 'kh',  '\u0c17': 'g',   '\u0c18': 'gh',  '\u0c19': 'ng',
    '\u0c1a': 'c',   '\u0c1b': 'ch',  '\u0c1c': 'j',   '\u0c1d': 'jh',  '\u0c1e': 'ny',
    '\u0c1f': 'tt',  '\u0c20': 'tth', '\u0c21': 'dd',  '\u0c22': 'ddh', '\u0c23': 'nn',
    '\u0c24': 't',   '\u0c25': 'th',  '\u0c26': 'd',   '\u0c27': 'dh',  '\u0c28': 'n',
    '\u0c2a': 'p',   '\u0c2b': 'ph',  '\u0c2c': 'b',   '\u0c2d': 'bh',  '\u0c2e': 'm',
    '\u0c2f': 'y',   '\u0c30': 'r',   '\u0c31': 'rr',  '\u0c32': 'l',   '\u0c33': 'll',
    '\u0c35': 'v',   '\u0c36': 'sh',  '\u0c37': 'ss',  '\u0c38': 's',   '\u0c39': 'h'
}
VIRAMA = '\u0c4d'
ANUSVARA = '\u0c02'
VISARGA = '\u0c03'


def telugu_to_tokens(text):
    text = unicodedata.normalize('NFC', text)
    words = re.split(r'(\s+)', text)
    all_tokens = []
    for w in words:
        if not w:
            continue
        if re.match(r'^\s+$', w):
            if len(w) > 1:
                all_tokens.append('@sp')
            continue
        chars = list(w)
        i = 0
        n = len(chars)
        while i < n:
            ch = chars[i]
            if ch in TELUGU_VOWELS:
                for c in TELUGU_VOWELS[ch]:
                    all_tokens.append('@' + c)
                i += 1
            elif ch in TELUGU_CONSONANTS:
                base_c = TELUGU_CONSONANTS[ch]
                if i + 1 < n:
                    nxt = chars[i+1]
                    if nxt == VIRAMA:
                        for c in base_c:
                            all_tokens.append('@' + c)
                        i += 2
                    elif nxt in TELUGU_MATRAS:
                        for c in base_c:
                            all_tokens.append('@' + c)
                        for m in TELUGU_MATRAS[nxt]:
                            all_tokens.append('@' + m)
                        i += 2
                    else:
                        for c in base_c:
                            all_tokens.append('@' + c)
                        all_tokens.append('@a')
                        i += 1
                else:
                    for c in base_c:
                        all_tokens.append('@' + c)
                    all_tokens.append('@a')
                    i += 1
            elif ch == ANUSVARA:
                all_tokens.append('@m')
                i += 1
            elif ch == VISARGA:
                all_tokens.append('@h')
                i += 1
            else:
                i += 1
    return all_tokens


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
        wav = vocoder(mel_predicted).squeeze().cpu().numpy()
        return wav, output[1][0].cpu().numpy()


def main():
    sys.argv = [arg for arg in sys.argv if arg != "\\"]
    parser = argparse.ArgumentParser(description="Telugu FastSpeech 2 Speech Synthesis")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint file (.pth)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--text", type=str, default=None, help="Telugu text to synthesize")
    parser.add_argument("--text-file", "--text_file", type=str, default=None, help="Path to text file containing Telugu text")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker integer ID (0: female, 1: male)")
    parser.add_argument("--output_wav", type=str, default="telugu_synthesized.wav", help="Destination WAV file")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.40, help="Duration control (default: 1.40 for calibrated pacing)")
    parser.add_argument("--test_all", action="store_true", help="Synthesize all official assignment benchmark test sentences")
    args = parser.parse_args()

    if args.test_all:
        try:
            import generate_telugu_assignment_audios
            print("Running full official Telugu benchmark synthesis...")
            generate_telugu_assignment_audios.main()
            return
        except Exception as e:
            print(f"Error running benchmark generator: {e}")
            return

    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            args.text = f.read().strip()

    if not args.text:
        args.text = "నమస్కారం"

    config_path = args.config
    if not os.path.exists(config_path):
        candidate = os.path.join(PACKAGE_ROOT, "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            candidate2 = os.path.join(PACKAGE_ROOT, "Telugu", "config.yaml")
            if os.path.exists(candidate2):
                config_path = candidate2

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Fallback to pre-bundled stats.json and speakers.json in repository if preprocessed_data is absent
    prep_path = cfg["path"]["preprocessed_path"]
    if not os.path.isabs(prep_path):
        prep_path = os.path.join(PACKAGE_ROOT, prep_path)
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
            "/kaggle/working/Telugu/checkpoints/best_model.pth",
            "/kaggle/working/Telugu/checkpoints/latest_checkpoint.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {args.checkpoint} or common locations.")
        print("Please download best_model.pth from Google Drive to Telugu/checkpoints/best_model.pth")
        sys.exit(1)

    configs = (
        {"dataset": cfg["dataset"], "path": cfg["path"], "preprocessing": cfg["preprocessing"]},
        cfg["model"],
        cfg["train"],
    )

    cleaners = configs[0]["preprocessing"]["text"]["text_cleaners"]
    tokens = telugu_to_tokens(args.text)
    phone_str = "{" + " ".join(tokens) + "}"
    seq = text_to_sequence(phone_str, cleaners)

    print(f"Loading model checkpoint from {ckpt_path}...")
    model = get_model(args, configs, device, train=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    print("Loading HiFi-GAN vocoder...")
    vocoder = get_vocoder(configs[1], device)

    print(f"Synthesizing Telugu text: {args.text}")
    print(f"Token sequence: {phone_str}")
    wav, mel = synthesize_utterance(
        model,
        vocoder,
        seq,
        args.speaker_id,
        configs,
        control_values=(args.pitch_control, args.energy_control, args.duration_control),
    )

    sr = configs[0]["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(args.output_wav, sr, (wav * 32767).clip(-32767, 32767).astype(np.int16))
    print(f"Successfully saved synthesized audio to: {args.output_wav} ({len(wav) / sr:.2f} seconds)")


if __name__ == "__main__":
    main()

