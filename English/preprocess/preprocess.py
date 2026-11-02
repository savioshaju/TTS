"""
English TTS Preprocessing Module for Svarah Dataset.
Implements Stage B: Text Processing, Normalization, MFA Forced Alignment,
and TextGrid Validation following the FastSpeech 2 methodology.
"""
import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
if DIR not in sys.path:
    sys.path.insert(0, DIR)
import argparse
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "FastSpeech2-master"))

import stage_and_align_english as sae

def preprocess_english(config_path, stage="all", num_jobs=10):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    splits = ["val", "train"]
    reports = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing English Svarah split: {split.upper()}")
        print(f"{'='*60}")

        # 1. Stage corpus
        input_dir, records = sae.stage_corpus(split)

        # 2. MFA Forced alignment
        output_dir = os.path.join(ROOT_DIR, "preprocessing", "English", "TextGrid", split)
        if stage in ["all", "align"]:
            elapsed = sae.run_mfa_alignment(input_dir, output_dir, num_jobs=num_jobs)

        # 3. TextGrid conversion and validation
        if stage in ["all", "convert"]:
            report = sae.convert_and_validate(split, records, output_dir)
            reports[split] = report

    return reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Svarah English dataset for FastSpeech 2")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--stage", type=str, default="all", choices=["all", "align", "convert"])
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()

    preprocess_english(args.config, stage=args.stage, num_jobs=args.jobs)

