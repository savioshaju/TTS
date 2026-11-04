"""
Custom Synthesis Script for Telugu FastSpeech 2
Supports arbitrary Telugu text input via native Unicode akshara tokenization.
"""

import os
import sys
import re
import argparse
import yaml
import torch
import numpy as np
import soundfile as sf
import unicodedata
from scipy.signal import resample_poly
from math import gcd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "Telugu":
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BASE_DIR = SCRIPT_DIR

PKG_DIR = os.path.join(SCRIPT_DIR, "inference")
if not os.path.exists(PKG_DIR):
    PKG_DIR = os.path.join(BASE_DIR, "kaggle_staging", "Telugu_FastSpeech2_Kaggle")
if not os.path.exists(PKG_DIR):
    PKG_DIR = os.path.join(BASE_DIR, "FastSpeech2-master")

if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import FastSpeech2
from utils.model import get_vocoder
from utils.tools import get_mask_from_lengths
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

TELUGU_VOWEL_TOKENS = {
    '@a', '@aa', '@i', '@ii', '@u', '@uu', '@ru', '@e', '@ee', '@ai', '@o', '@oo', '@au'
}

TELUGU_NUMERALS_MAP = str.maketrans('౦౧౨౩౪౫౬౭౮౯', '0123456789')

ONES_TE = {
    0: "సున్నా", 1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు",
    5: "ఐదు", 6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది"
}

TEENS_TE = {
    10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు",
    15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పందొమ్మిది"
}

TENS_TE = {
    20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై",
    60: "అరవై", 70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై"
}

def number_to_telugu_words(n: int) -> str:
    """
    Recursively converts an integer n (0 to 999,999,999) into spoken Telugu words.
    """
    if n < 0:
        return "మైనస్ " + number_to_telugu_words(-n)
    if n < 10:
        return ONES_TE[n]
    if n < 20:
        return TEENS_TE[n]
    if n < 100:
        ten = (n // 10) * 10
        rem = n % 10
        if rem == 0:
            return TENS_TE[ten]
        else:
            return f"{TENS_TE[ten]} {ONES_TE[rem]}"
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        h_word = "వంద" if hundreds == 1 else f"{ONES_TE[hundreds]} వందలు"
        if rem == 0:
            return h_word
        else:
            prefix = "వంద" if hundreds == 1 else f"{ONES_TE[hundreds]} వందల"
            return f"{prefix} {number_to_telugu_words(rem)}"
    if n < 100000:
        thousands = n // 1000
        rem = n % 1000
        th_word = "వెయ్యి" if thousands == 1 else f"{number_to_telugu_words(thousands)} వేలు"
        if rem == 0:
            return th_word
        else:
            prefix = "వెయ్యి" if thousands == 1 else f"{number_to_telugu_words(thousands)} వేల"
            return f"{prefix} {number_to_telugu_words(rem)}"
    if n < 10000000:
        lakhs = n // 100000
        rem = n % 100000
        l_word = "లక్ష" if lakhs == 1 else f"{number_to_telugu_words(lakhs)} లక్షలు"
        if rem == 0:
            return l_word
        else:
            prefix = "లక్ష" if lakhs == 1 else f"{number_to_telugu_words(lakhs)} లక్షల"
            return f"{prefix} {number_to_telugu_words(rem)}"
    if n < 1000000000:
        crores = n // 10000000
        rem = n % 10000000
        c_word = "కోటి" if crores == 1 else f"{number_to_telugu_words(crores)} కోట్లు"
        if rem == 0:
            return c_word
        else:
            prefix = "కోటి" if crores == 1 else f"{number_to_telugu_words(crores)} కోట్ల"
            return f"{prefix} {number_to_telugu_words(rem)}"
    return str(n)

ORDINALS_SPECIAL_TE = {
    1: "ఒకటవ",
    2: "రెండవ",
    3: "మూడవ",
    4: "నాలుగవ",
    5: "ఐదవ",
    6: "ఆరవ",
    7: "ఏడవ",
    8: "ఎనిమిదవ",
    9: "తొమ్మిదవ",
    10: "పదవ",
    11: "పదకొండవ",
    12: "పన్నెండవ",
}

def ordinal_to_telugu(n: int) -> str:
    if n in ORDINALS_SPECIAL_TE:
        return ORDINALS_SPECIAL_TE[n]
    words = number_to_telugu_words(n)
    if words.endswith("ి") or words.endswith("ు") or words.endswith("ల"):
        return words + "వ"
    elif words.endswith("డు"):
        return words[:-2] + "డవ"
    return words + "వ"

def expand_numbers_te(text: str) -> str:
    """
    Normalizes numbers in Telugu text:
    - Converts Telugu Unicode digits (౦-౯) to ASCII digits.
    - Removes thousands commas (e.g. 10,000 -> 10000).
    - Expands time formats (e.g. 10:45 -> పది నలభై ఐదు).
    - Expands ordinals (e.g. 12వ -> పన్నెండవ).
    - Expands decimals (e.g. 3.14 -> మూడు పాయింట్ ఒకటి నాలుగు).
    - Expands standalone integers (e.g. 2026 -> రెండు వేల ఇరవై ఆరు).
    """
    text = text.translate(TELUGU_NUMERALS_MAP)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)

    # 1. Time formats e.g. 10:45, 09:30
    def _expand_time(m):
        h = int(m.group(1))
        mins = int(m.group(2))
        h_words = number_to_telugu_words(h)
        if mins == 0:
            return f" {h_words} "
        m_words = number_to_telugu_words(mins)
        return f" {h_words} {m_words} "
    text = re.sub(r'\b([0-2]?[0-9]):([0-5][0-9])\b', _expand_time, text)

    # 2. Ordinals e.g. 12వ, 12 వ
    def _expand_ordinal(m):
        n = int(m.group(1))
        return f" {ordinal_to_telugu(n)} "
    text = re.sub(r'\b([0-9]+)\s*వ\b', _expand_ordinal, text)

    # 3. Decimals e.g. 3.14 -> మూడు పాయింట్ ఒకటి నాలుగు
    def _expand_decimal(m):
        integer_part, frac_part = m.group(1), m.group(2)
        int_words = number_to_telugu_words(int(integer_part))
        frac_words = " ".join(ONES_TE.get(int(d), d) for d in frac_part)
        return f" {int_words} పాయింట్ {frac_words} "
    text = re.sub(r'(\d+)\.(\d+)', _expand_decimal, text)

    # 4. Standalone integers
    def _expand_int(m):
        return f" {number_to_telugu_words(int(m.group(0)))} "
    text = re.sub(r'\b[0-9]+\b', _expand_int, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def telugu_to_tokens(text):
    text = expand_numbers_te(text)
    text = unicodedata.normalize('NFC', text)
    words = re.split(r'(\s+|[.,!?;])', text)
    all_tokens = []
    for w in words:
        if not w:
            continue
        if w in [",", ";", ".", "!", "?"]:
            all_tokens.append('@sp')
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

def main():
    parser = argparse.ArgumentParser(description="FastSpeech 2 Telugu Custom Text Synthesizer")
    parser.add_argument("--text", type=str, default="నమస్కారం, మీరు ఎలా ఉన్నారు?", help="Telugu text to synthesize")
    parser.add_argument("--phonemes", type=str, default=None, help="Explicit phoneme string (overrides text)")
    parser.add_argument("--output", type=str, default="custom_synthesize/telugu_custom.wav", help="Output WAV path")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID (default 0 = telugufemale)")
    parser.add_argument("--d_control", type=float, default=1.25, help="Duration control speed factor (default: 1.25 for calibrated clarity)")
    parser.add_argument(
        "--mode",
        type=str,
        default="duration_improved",
        choices=["baseline", "duration_improved"],
        help="Inference clarity mode: 'duration_improved' (default, vowel-protected) or 'baseline'",
    )
    args = parser.parse_args()

    # 1. Phonemes
    if args.phonemes:
        phone_str = args.phonemes
        print(f"Using provided phonemes: {phone_str}")
    else:
        tokens = telugu_to_tokens(args.text)
        phone_str = "{" + " ".join(tokens) + "}"
        print(f"Input text: '{args.text}'")
        print(f"Mode: {args.mode}")
        print(f"Derived phonemes: {phone_str}")

    # 2. Config & Model
    cfg_path = os.path.join(SCRIPT_DIR, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Locate stats.json directory (self-contained fallback)
    stats_candidates = [
        SCRIPT_DIR,
        os.path.join(SCRIPT_DIR, "inference"),
        os.path.join(BASE_DIR, "preprocessing", "Telugu"),
    ]
    stats_dir = SCRIPT_DIR
    for d in stats_candidates:
        if os.path.exists(os.path.join(d, "stats.json")):
            stats_dir = d
            break

    cfg["path"]["preprocessed_path"] = stats_dir
    preprocess_config = {
        "dataset": cfg["dataset"],
        "path": cfg["path"],
        "preprocessing": cfg["preprocessing"],
    }
    model_config = cfg["model"]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    print("Loading Telugu FastSpeech 2 Model...")
    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt_path = os.path.join(BASE_DIR, "Telugu", "checkpoints", "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    print("Loading HiFi-GAN Vocoder...")
    vocoder = get_vocoder(model_config, device)

    # 3. Inference with calibrated duration control
    tokens = [p for p in phone_str.strip("{}").split() if p]
    seq = np.array(text_to_sequence(phone_str, cleaners))
    txt_tensor = torch.from_numpy(np.array([seq])).long().to(device)
    txt_lens = torch.from_numpy(np.array([len(seq)])).long().to(device)
    spk_tensor = torch.from_numpy(np.array([args.speaker_id])).long().to(device)
    max_len = max(txt_lens).item()

    with torch.no_grad():
        src_mask = get_mask_from_lengths(txt_lens, max_len)
        output = model.encoder(txt_tensor, src_mask)
        if model.speaker_emb is not None:
            output = output + model.speaker_emb(spk_tensor).unsqueeze(1).expand(-1, max_len, -1)
            
        # Duration prediction based on selectable mode
        log_d = model.variance_adaptor.duration_predictor(output, src_mask)
        d_raw = torch.clamp(torch.exp(log_d) - 1, min=0).squeeze(0)
        
        if args.mode == "baseline":
            d_base = torch.round(d_raw)
            d_scaled = torch.clamp(torch.round(d_base * args.d_control), min=0)
        else:
            # Calibrated duration regulation + selective vowel nucleus protection
            d_scaled = torch.round(d_raw * args.d_control)
            for idx, tok in enumerate(tokens):
                if tok in TELUGU_VOWEL_TOKENS and d_scaled[idx] < 2.0:
                    d_scaled[idx] = 2.0
                    
        d_scaled_batch = d_scaled.unsqueeze(0)
        
        # Pitch and energy
        _, pitch_emb = model.variance_adaptor.get_pitch_embedding(output, None, src_mask, 1.0)
        output = output + pitch_emb
        _, energy_emb = model.variance_adaptor.get_energy_embedding(output, None, src_mask, 1.0)
        output = output + energy_emb
        
        # Length regulation & decoder
        output, mel_len = model.variance_adaptor.length_regulator(output, d_scaled_batch, None)
        mel_mask = get_mask_from_lengths(mel_len)
        mel_output, mel_mask = model.decoder(output, mel_mask)
        mel_output = model.mel_linear(mel_output)
        postnet_output = model.postnet(mel_output) + mel_output
        mel_pred = postnet_output.transpose(1, 2)
        
        # Vocoder generation & 16kHz resampling
        wav = vocoder(mel_pred).squeeze(1).cpu().numpy()[0]
        g = gcd(16000, 22050)
        wav_16k = resample_poly(wav, 16000 // g, 22050 // g)
        wav_16k = np.clip(wav_16k, -1.0, 1.0)

        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, wav_16k, 16000)
        print(f"\n[SUCCESS] Telugu audio generated: {out_path}")
        print(f"  Duration: {len(wav_16k)/16000:.2f} seconds")
        print(f"  Sample Rate: 16000 Hz")

if __name__ == "__main__":
    main()

