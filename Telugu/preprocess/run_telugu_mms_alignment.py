"""
Stage C — Full-Dataset Meta MMS Forced Alignment for Telugu FastSpeech 2.
Aligns all 2,695 resampled utterances (2,250 train, 445 val) using torchaudio.pipelines.MMS_FA on GPU.
- Applies Unicode NFC normalization.
- Remediates Latin 'o' typo for Telugu Anusvara (U+0C02).
- Extracts word-level and token-level boundaries and confidence scores.
- Saves individual alignments under D:\\Savio Shaju\\TTS\\preprocessing\\Telugu\\alignments\\{uid}.json.
- Logs failure list and full alignment metrics.
"""

import os
import sys
import wave
import json
import time
import unicodedata
import re
import numpy as np
import torch
import torchaudio

ROOT_DIR = r"D:\Savio Shaju\TTS"
WAVS_DIR = os.path.join(ROOT_DIR, "preprocessing", "Telugu", "wavs_16k")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "prepared", "telugu_prepared_v2")
ALIGN_DIR = os.path.join(ROOT_DIR, "preprocessing", "Telugu", "alignments")
REPORT_DIR = os.path.join(ROOT_DIR, "Report")

os.makedirs(ALIGN_DIR, exist_ok=True)

print("=" * 75)
print("STAGE C: FULL-DATASET META MMS FORCED ALIGNMENT (TELUGU)")
print(f"Audio Source:      {WAVS_DIR}")
print(f"Alignment Output:  {ALIGN_DIR}")
print("=" * 75)

# 1. Telugu G2P & Text Normalization
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

def telugu_normalize_and_romanize(text):
    """
    Applies Unicode NFC normalization, repairs Latin 'o' typo for Anusvara,
    and maps Telugu script into romanized tokens compatible with MMS_FA.
    """
    text = unicodedata.normalize('NFC', text)
    # Fix typist typo: Latin 'o' used for Telugu Anusvara 'ం' (U+0C02)
    text = re.sub(r'(?<=[\u0c00-\u0c7f])o(?=[\u0c00-\u0c7f\s\.,!?]|$)', ANUSVARA, text)
    words = re.findall(r"[\u0c00-\u0c7f]+", text)
    
    roman_words = []
    ipa_phones = []
    
    for word in words:
        w_rom = []
        w_ipa = []
        i = 0
        n = len(word)
        while i < n:
            ch = word[i]
            if ch in TELUGU_VOWELS:
                v = TELUGU_VOWELS[ch]
                w_rom.append(v)
                w_ipa.append(v)
                i += 1
            elif ch in TELUGU_CONSONANTS:
                c = TELUGU_CONSONANTS[ch]
                if i + 1 < n:
                    nxt = word[i+1]
                    if nxt == VIRAMA:
                        w_rom.append(c)
                        w_ipa.append(c)
                        i += 2
                    elif nxt in TELUGU_MATRAS:
                        m = TELUGU_MATRAS[nxt]
                        w_rom.append(c + m)
                        w_ipa.extend([c, m])
                        i += 2
                    else:
                        w_rom.append(c + 'a')
                        w_ipa.extend([c, 'a'])
                        i += 1
                else:
                    w_rom.append(c + 'a')
                    w_ipa.extend([c, 'a'])
                    i += 1
            elif ch == ANUSVARA:
                w_rom.append('m')
                w_ipa.append('m')
                i += 1
            elif ch == VISARGA:
                w_rom.append('h')
                w_ipa.append('h')
                i += 1
            else:
                i += 1
        if w_rom:
            roman_words.append("".join(w_rom))
            ipa_phones.extend(w_ipa)
            
    return roman_words, ipa_phones

# 2. Load Meta MMS Forced Aligner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing Meta MMS Forced Aligner on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model().to(device)
model.eval()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
dict_map = bundle.get_dict()
print(f"MMS_FA initialized successfully. Vocabulary dictionary size: {len(dict_map)} tokens.")

# 3. Read All Utterance Metadata
records = []
for split in ["train", "val"]:
    meta_path = os.path.join(DATASET_DIR, split, "metadata.csv")
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("|")
            uid, spk, text, gender, dur = p[0], p[1], p[2], p[3], float(p[4])
            wav_path = os.path.join(WAVS_DIR, f"{uid}.wav")
            records.append({
                "uid": uid,
                "split": split,
                "speaker": spk,
                "gender": gender,
                "text": text,
                "duration": dur,
                "wav_path": wav_path,
            })

print(f"Loaded {len(records):,} records to align (Train: 2,250, Val: 445). Starting batch alignment...")

start_time = time.time()
success_count = 0
failed_count = 0
failed_utterances = []
all_results = []
all_scores = []
female_scores = []
male_scores = []

frame_dur = 320.0 / 16000.0  # 20.0 ms per MMS emission frame

for idx, rec in enumerate(records):
    uid = rec["uid"]
    wav_path = rec["wav_path"]
    text = rec["text"]
    spk = rec["speaker"]
    gender = rec["gender"]
    dur = rec["duration"]

    # Read 16 kHz audio
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        nf = wf.getnframes()
        frames = wf.readframes(nf)
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)

    # Normalize and romanize
    roman_words, ipa_phones = telugu_normalize_and_romanize(text)
    clean_words = []
    for w in roman_words:
        cw = "".join(c for c in w.lower() if c in dict_map)
        if cw:
            clean_words.append(cw)

    if not clean_words:
        failed_count += 1
        failed_utterances.append({"uid": uid, "reason": "No valid romanized words after filtering"})
        continue

    tokens = tokenizer(clean_words)

    try:
        with torch.no_grad():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
            token_spans = aligner(emissions[0], tokens)

        word_alignments = []
        utt_scores = []

        for w_str, spans in zip(clean_words, token_spans):
            w_start = round(spans[0].start * frame_dur, 3)
            w_end = round(spans[-1].end * frame_dur, 3)
            w_dur = round(w_end - w_start, 3)
            char_aligns = []
            for c_ch, sp in zip(w_str, spans):
                c_start = round(sp.start * frame_dur, 3)
                c_end = round(sp.end * frame_dur, 3)
                sc = round(float(sp.score), 4)
                utt_scores.append(sp.score)
                char_aligns.append({
                    "char": c_ch,
                    "start": c_start,
                    "end": c_end,
                    "dur": round(c_end - c_start, 3),
                    "score": sc,
                })
            word_alignments.append({
                "word": w_str,
                "start": w_start,
                "end": w_end,
                "dur": w_dur,
                "chars": char_aligns,
            })

        mean_score = round(float(np.mean(utt_scores)), 4) if utt_scores else 0.0
        success_count += 1
        all_scores.append(mean_score)
        if gender == "Female":
            female_scores.append(mean_score)
        else:
            male_scores.append(mean_score)

        # Save individual alignment JSON
        align_entry = {
            "uid": uid,
            "speaker": spk,
            "gender": gender,
            "split": rec["split"],
            "total_duration": dur,
            "word_count": len(clean_words),
            "phone_count": len(ipa_phones),
            "mean_confidence": mean_score,
            "words": word_alignments,
        }
        with open(os.path.join(ALIGN_DIR, f"{uid}.json"), "w", encoding="utf-8") as f_out:
            json.dump(align_entry, f_out, indent=2)

        all_results.append({
            "uid": uid,
            "speaker": spk,
            "gender": gender,
            "split": rec["split"],
            "duration": dur,
            "word_count": len(clean_words),
            "mean_confidence": mean_score,
            "success": True,
        })

    except Exception as e:
        failed_count += 1
        failed_utterances.append({"uid": uid, "reason": str(e), "text": text[:60]})
        all_results.append({
            "uid": uid,
            "speaker": spk,
            "gender": gender,
            "split": rec["split"],
            "duration": dur,
            "success": False,
            "error": str(e),
        })

    if (idx + 1) % 300 == 0 or (idx + 1) == len(records):
        elapsed = time.time() - start_time
        fps = (idx + 1) / elapsed
        print(f"  Aligned {idx + 1:,} / {len(records):,} utterances ({elapsed:.1f}s, {fps:.1f} utt/s, successes: {success_count:,}, fails: {failed_count})...")

total_elapsed = time.time() - start_time
print("\n" + "=" * 75)
print("STAGE C: MMS FORCED ALIGNMENT RUN COMPLETE")
print(f"  Total Processed:         {len(records):,} utterances")
print(f"  Alignment Successes:     {success_count:,} ({success_count / len(records) * 100:.2f}%)")
print(f"  Alignment Failures:      {failed_count:,} ({failed_count / len(records) * 100:.2f}%)")
print(f"  Total Run Time:          {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
print(f"  Overall Mean Confidence: {np.mean(all_scores):.4f} +/- {np.std(all_scores):.4f}")
print(f"  Female Mean Confidence:  {np.mean(female_scores):.4f} +/- {np.std(female_scores):.4f} (n={len(female_scores):,})")
print(f"  Male Mean Confidence:    {np.mean(male_scores):.4f} +/- {np.std(male_scores):.4f} (n={len(male_scores):,})")
print("=" * 75)

# Save alignment summary and failure logs
summary_data = {
    "stage": "Stage C — Telugu MMS Forced Alignment",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "total_processed": len(records),
    "success_count": success_count,
    "failure_count": failed_count,
    "success_rate_percent": round(success_count / len(records) * 100, 2),
    "elapsed_seconds": round(total_elapsed, 2),
    "confidence_stats": {
        "overall_mean": round(float(np.mean(all_scores)), 4),
        "overall_std": round(float(np.std(all_scores)), 4),
        "overall_min": round(float(np.min(all_scores)), 4),
        "overall_max": round(float(np.max(all_scores)), 4),
        "female_mean": round(float(np.mean(female_scores)), 4),
        "female_std": round(float(np.std(female_scores)), 4),
        "male_mean": round(float(np.mean(male_scores)), 4),
        "male_std": round(float(np.std(male_scores)), 4),
    },
    "failed_utterances": failed_utterances,
}

with open(os.path.join(REPORT_DIR, "telugu_stage_c_alignment_results.json"), "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

print(f"Full alignment summary saved to: Report/telugu_stage_c_alignment_results.json")

