import os
import sys
import subprocess
import json
import time
import shutil
import numpy as np
import soundfile as sf
from praatio import textgrid

# Add FastSpeech2 to path for text cleaning and symbols
sys.path.append(r"D:\Savio Shaju\TTS\FastSpeech2-master")
from text import _clean_text
from text.symbols import symbols

SAMPLING_RATE = 16000
HOP_LENGTH = 256
FPS = SAMPLING_RATE / HOP_LENGTH  # 62.5

MFA_ENV_PATH = r"D:\Savio Shaju\TTS\.mfa_env"
MFA_ROOT_DIR = r"D:\Savio Shaju\TTS\mfa_root"
PYTHON_EXE = os.path.join(MFA_ENV_PATH, "python.exe")

PREPROCESSING_BASE = r"D:\Savio Shaju\TTS\preprocessing\English"
DATASET_BASE = r"D:\Savio Shaju\TTS\datasets\prepared\english_speaker_independent"

def get_env():
    env = os.environ.copy()
    env["PATH"] = (
        os.path.join(MFA_ENV_PATH, "Library", "bin") + ";" +
        os.path.join(MFA_ENV_PATH, "Scripts") + ";" +
        MFA_ENV_PATH + ";" +
        env.get("PATH", "")
    )
    env["MFA_ROOT_DIR"] = MFA_ROOT_DIR
    env["PYTHONIOENCODING"] = "utf-8"
    return env

def stage_corpus(split_name):
    metadata_path = os.path.join(DATASET_BASE, split_name, "metadata.csv")
    wavs_dir = os.path.join(DATASET_BASE, split_name, "wavs")
    staged_input_dir = os.path.join(PREPROCESSING_BASE, "mfa_input", split_name)
    
    os.makedirs(staged_input_dir, exist_ok=True)
    print(f"Staging {split_name} corpus into {staged_input_dir}...")
    
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 7:
                records.append({
                    "basename": parts[0],
                    "speaker": parts[1],
                    "raw_text": parts[2],
                    "gender": parts[3],
                    "language": parts[4],
                    "district": parts[5],
                    "duration": float(parts[6])
                })
                
    staged_count = 0
    for rec in records:
        spk = rec["speaker"]
        base = rec["basename"]
        spk_dir = os.path.join(staged_input_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        
        src_wav = os.path.join(wavs_dir, f"{base}.wav")
        dst_wav = os.path.join(spk_dir, f"{base}.wav")
        dst_lab = os.path.join(spk_dir, f"{base}.lab")
        
        # Hard link wav if not already linked
        if not os.path.exists(dst_wav):
            try:
                os.link(src_wav, dst_wav)
            except Exception:
                shutil.copyfile(src_wav, dst_wav)
                
        # Clean text and write lab
        cleaned = _clean_text(rec["raw_text"], ["english_cleaners"])
        with open(dst_lab, "w", encoding="utf-8") as f_lab:
            f_lab.write(cleaned.strip())
            
        staged_count += 1
        
    print(f"Successfully staged {staged_count} utterances across {len(set(r['speaker'] for r in records))} speakers for {split_name}.")
    return staged_input_dir, records

def run_mfa_alignment(input_dir, output_dir, num_jobs=8):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        PYTHON_EXE,
        "-m", "montreal_forced_aligner.command_line.mfa",
        "align",
        input_dir,
        "english_india_mfa",
        "english_mfa",
        output_dir,
        "--clean",
        "--overwrite",
        "-j", str(num_jobs)
    ]
    
    print(f"\nRunning MFA alignment command:\n{' '.join(cmd)}")
    start_t = time.time()
    
    env = get_env()
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    elapsed = time.time() - start_t
    print(f"MFA execution finished in {elapsed:.2f} seconds (exit code: {proc.returncode}).")
    if proc.returncode != 0:
        print("MFA Output:\n", proc.stdout[-2000:])
        raise RuntimeError(f"MFA failed with code {proc.returncode}")
    else:
        # Print summary lines from stdout
        for line in proc.stdout.splitlines()[-15:]:
            print("  ", line)
    return elapsed

def convert_and_validate(split_name, records, textgrid_dir):
    print(f"\nConverting TextGrids to mel-frame durations for {split_name}...")
    
    duration_dir = os.path.join(PREPROCESSING_BASE, "duration")
    os.makedirs(duration_dir, exist_ok=True)
    
    sil_phones = {"sil", "sp", "spn", ""}
    symbol_set = set(symbols)
    
    aligned_records = []
    unaligned_records = []
    all_phones = []
    durations_list = []
    spn_count = 0
    total_phone_count = 0
    total_audio_frames = 0
    total_duration_frames = 0
    
    metadata_entries = []
    
    for rec in records:
        spk = rec["speaker"]
        base = rec["basename"]
        raw_text = rec["raw_text"]
        
        tg_path = os.path.join(textgrid_dir, spk, f"{base}.TextGrid")
        if not os.path.exists(tg_path):
            # Fallback if flat
            tg_path = os.path.join(textgrid_dir, f"{base}.TextGrid")
            
        if not os.path.exists(tg_path):
            unaligned_records.append(base)
            continue
            
        try:
            tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
            phone_tier = tg.getTier("phones")
            
            phones = []
            durations = []
            start_time = 0
            end_time = 0
            end_idx = 0
            
            for entry in phone_tier.entries:
                s, e, p = entry.start, entry.end, entry.label.strip()
                if not phones:
                    if p in sil_phones or not p:
                        continue
                    else:
                        start_time = s
                        
                if p not in sil_phones and p:
                    phones.append(p)
                    dur = max(1, int(np.round(e * FPS) - np.round(s * FPS)))
                    durations.append(dur)
                    end_time = e
                    end_idx = len(phones)
                else:
                    # Internal pause / silence
                    pause_phone = "spn" if p == "spn" else "sp"
                    dur = int(np.round(e * FPS) - np.round(s * FPS))
                    if dur > 0:
                        phones.append(pause_phone)
                        durations.append(dur)
                    
            trimmed_phones = phones[:end_idx]
            trimmed_durations = durations[:end_idx]
            
            if not trimmed_phones or sum(trimmed_durations) <= 0:
                unaligned_records.append(base)
                continue
                
            # Check symbols compatibility
            for p in trimmed_phones:
                sym = "@" + p
                if sym not in symbol_set and p not in symbol_set:
                    print(f"Warning: phoneme {p} (symbol {sym}) not in FastSpeech2 symbols!")
                    
            # Save duration array
            dur_filename = f"{spk}-duration-{base}.npy"
            dur_path = os.path.join(duration_dir, dur_filename)
            np.save(dur_path, np.array(trimmed_durations, dtype=np.int32))
            
            # Format metadata entry: basename|speaker|{phones}|raw_text
            phone_str = "{" + " ".join(trimmed_phones) + "}"
            metadata_entries.append(f"{base}|{spk}|{phone_str}|{raw_text}")
            
            aligned_records.append(base)
            durations_list.extend(trimmed_durations)
            all_phones.extend(trimmed_phones)
            spn_count += sum(1 for p in trimmed_phones if p == "spn")
            total_phone_count += len(trimmed_phones)
            total_duration_frames += sum(trimmed_durations)
            total_audio_frames += int(np.round(rec["duration"] * FPS))
            
        except Exception as ex:
            print(f"Error processing {tg_path}: {ex}")
            unaligned_records.append(base)
            
    # Write metadata txt file (FastSpeech 2 train.txt / val.txt)
    meta_txt_path = os.path.join(PREPROCESSING_BASE, f"{split_name}.txt")
    with open(meta_txt_path, "w", encoding="utf-8") as f_txt:
        for entry in metadata_entries:
            f_txt.write(entry + "\n")
            
    dur_arr = np.array(durations_list) if durations_list else np.array([0])
    
    report = {
        "split": split_name,
        "total_staged": len(records),
        "aligned_count": len(aligned_records),
        "unaligned_count": len(unaligned_records),
        "alignment_rate_pct": round(len(aligned_records) / len(records) * 100, 2),
        "total_phonemes": total_phone_count,
        "unique_phonemes": len(set(all_phones)),
        "spn_phonemes": spn_count,
        "spn_rate_pct": round(spn_count / max(1, total_phone_count) * 100, 2),
        "duration_mel_frames": {
            "total_frames": int(total_duration_frames),
            "total_audio_frames": int(total_audio_frames),
            "min": int(dur_arr.min()),
            "max": int(dur_arr.max()),
            "mean": round(float(dur_arr.mean()), 2),
            "median": float(np.median(dur_arr)),
            "std": round(float(dur_arr.std()), 2)
        },
        "mean_phoneme_duration_ms": round(float(dur_arr.mean() * HOP_LENGTH / SAMPLING_RATE * 1000), 2)
    }
    
    report_path = os.path.join(PREPROCESSING_BASE, f"report_{split_name}.json")
    with open(report_path, "w", encoding="utf-8") as f_rep:
        json.dump(report, f_rep, indent=2)
        
    print(f"\n=== {split_name.upper()} VALIDATION SUMMARY ===")
    print(f"Aligned: {len(aligned_records)}/{len(records)} ({report['alignment_rate_pct']}%)")
    print(f"Phonemes: {total_phone_count} (Unique: {report['unique_phonemes']}, OOV/spn: {spn_count} [{report['spn_rate_pct']}%])")
    print(f"Phoneme duration (frames): mean={report['duration_mel_frames']['mean']}, median={report['duration_mel_frames']['median']}")
    print(f"Mean phoneme duration: {report['mean_phoneme_duration_ms']} ms")
    print(f"Report saved to: {report_path}")
    print(f"FastSpeech 2 metadata saved to: {meta_txt_path}")
    
    return report

if __name__ == "__main__":
    split_to_run = sys.argv[1] if len(sys.argv) > 1 else "val"
    num_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    input_dir, records = stage_corpus(split_to_run)
    output_dir = os.path.join(PREPROCESSING_BASE, "TextGrid", split_to_run)
    
    elapsed = run_mfa_alignment(input_dir, output_dir, num_jobs=num_jobs)
    report = convert_and_validate(split_to_run, records, output_dir)

