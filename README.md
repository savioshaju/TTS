# Multi-Speaker FastSpeech 2: English & Telugu Text-to-Speech

A robust, production-grade non-autoregressive Text-to-Speech (TTS) pipeline implementing **FastSpeech 2** for both **English (Svarah)** and **Telugu (IndicTTS)** with multi-speaker conditioning and HiFi-GAN neural vocoding at **16 kHz**.

---

## 1. Project Overview

FastSpeech 2 synthesizes mel-spectrograms non-autoregressively in parallel from text and phoneme sequences. The pipeline incorporates a **Variance Adaptor** that conditions phonetic representations on duration, continuous pitch ($F_0$), and energy, resolving the one-to-many mapping problem in acoustic modeling.

### Key Highlights
- **Bilingual & Multi-Speaker**:
  - **English (Svarah)**: Multi-speaker architecture conditioned on 104 Indian English speakers.
  - **Telugu (IndicTTS)**: High-fidelity bilingual model trained on male and female native speakers.
- **Phoneme-Level Forced Alignment**:
  - **English**: Montreal Forced Aligner (MFA) v2.2 with ARPAbet/IPA phone mapping.
  - **Telugu**: Meta MMS (Massively Multilingual Speech) CTC alignment with native *akshara* grapheme-to-phoneme conversion.
- **Acoustic Fidelity**:
  - 16,000 Hz audio sampling rate, 80-band mel-filterbank ($F_{\text{min}}=0\text{ Hz}, F_{\text{max}}=8000\text{ Hz}$, FFT size 1024, hop size 256).
  - Continuous pitch extraction via PyWorld with unvoiced frame linear interpolation.
  - Frame-wise L2-norm energy extraction.
- **Memory-Leak-Free Training Infrastructure**:
  - glibc heap trimming (`malloc_trim(0)`) prevents native C-heap fragmentation across 100,000 steps.
  - Dual Tesla T4 multi-GPU DataParallel scaling.
  - Strict checkpoint retention policy (Top 3 validation models + 1 latest checkpoint).
- **HiFi-GAN Universal Neural Vocoder**:
  - 16 kHz multi-period and multi-scale discriminator synthesis yielding artifact-free waveforms.

---

## 2. Repository Structure

```text
TTS/
├── English/
│   ├── train/
│   │   ├── train.py                  # Dual-GPU training engine with memory safeguards
│   │   ├── dataset.py                # PyTorch Dataset & dynamic batch collator
│   │   ├── evaluate.py               # Validation evaluation loop
│   │   ├── prepare_dataset.py        # Dataset verification and staging
│   │   ├── model/                    # FastSpeech 2, Variance Adaptor & Loss
│   │   ├── transformer/              # Multi-Head Attention, Feed-Forward Blocks
│   │   ├── audio/                    # Mel-spectrogram & STFT utilities
│   │   ├── text/                     # G2P, CMUDict, IPA symbols & cleaners
│   │   ├── utils/                    # Checkpoint management, tools, plotting
│   │   └── hifigan/                  # HiFi-GAN vocoder architecture & weights
│   ├── preprocess/
│   │   ├── preprocess.py             # Preprocessing pipeline orchestrator
│   │   ├── stage_and_align_english.py # MFA alignment automation
│   │   ├── extract_acoustic_features.py # Mel, continuous F0, energy extraction
│   │   ├── synchronize_english_manifests.py # Train/Val split synchronization
│   │   └── repartition_and_renormalize_speaker_dependent.py # Speaker stats
│   ├── inference/
│   │   ├── inference.py              # Single-sentence CLI inference
│   │   ├── generate_assignment_audios.py # Official benchmark generator
│   │   ├── generate_heteronym_comparison.py # Noun vs. verb heteronym testing
│   │   ├── compare_checkpoints_diagnostic.py # Checkpoint comparison
│   │   ├── model/, transformer/, audio/, text/, utils/ # Core inference modules
│   │   └── hifigan/                  # HiFi-GAN vocoder
│   ├── checkpoints/
│   │   ├── README.md                 # Checkpoint guide & Drive download links
│   │   └── checkpoints_index.json    # Retention metadata & loss rankings
│   ├── config.yaml                   # Acoustic, model, and training hyperparameters
│   └── English_FastSpeech2_Kaggle.ipynb # Plug-and-play Kaggle training notebook
│
├── Telugu/
│   ├── train/
│   │   ├── train.py                  # Dual-GPU training engine with memory safeguards
│   │   ├── dataset.py                # PyTorch Dataset & dynamic batch collator
│   │   ├── evaluate.py               # Validation evaluation loop
│   │   ├── prepare_dataset.py        # Dataset verification and staging
│   │   ├── train_safeguards.py       # Checkpoint & memory unit test safeguards
│   │   ├── model/                    # FastSpeech 2 architecture
│   │   ├── transformer/              # Multi-Head Attention, Feed-Forward Blocks
│   │   ├── audio/                    # Mel-spectrogram & STFT utilities
│   │   ├── text/                     # Telugu phonetic akshara symbol tables
│   │   ├── utils/                    # Checkpoint management, tools, plotting
│   │   └── hifigan/                  # HiFi-GAN vocoder architecture & weights
│   ├── preprocess/
│   │   ├── preprocess.py             # Preprocessing pipeline orchestrator
│   │   ├── run_telugu_mms_alignment.py # Meta MMS forced alignment
│   │   ├── extract_telugu_acoustic_features.py # Acoustic feature extraction
│   │   ├── resample_telugu_audio.py  # 16 kHz bandlimited polyphase resampling
│   │   └── create_telugu_splits.py   # Speaker-stratified manifests
│   ├── inference/
│   │   ├── inference.py              # Single-sentence CLI inference
│   │   ├── model/, transformer/, audio/, text/, utils/ # Core inference modules
│   │   └── hifigan/                  # HiFi-GAN vocoder
│   ├── checkpoints/
│   │   ├── README.md                 # Checkpoint guide & Drive download links
│   │   └── checkpoints_index.json    # Retention metadata & loss rankings
│   ├── config.yaml                   # Acoustic, model, and training hyperparameters
│   └── Telugu_FastSpeech2_Kaggle.ipynb # Plug-and-play Kaggle training notebook
│
├── Results/
│   ├── English/
│   │   ├── training_loss.png         # Training loss trajectory (Total, Mel, PostNet, Variance)
│   │   ├── validation_loss.png       # Validation loss trajectory
│   │   ├── generated_audios/         # Generated speech samples (E1.wav to E4.wav)
│   │   └── variance_adaptor_plots/
│   │       ├── duration/             # Duration prediction alignment plots
│   │       ├── pitch/                # F0 pitch contour tracking plots
│   │       └── energy/               # Energy contour tracking plots
│   └── Telugu/
│       ├── training_loss.png         # Training loss trajectory
│       ├── validation_loss.png       # Validation loss trajectory
│       ├── generated_audios/         # Generated speech samples (T1.wav to T4.wav)
│       └── variance_adaptor_plots/
│           ├── duration/             # Duration prediction alignment plots
│           ├── pitch/                # F0 pitch contour tracking plots
│           └── energy/               # Energy contour tracking plots
│
├── Report/
│   └── Technical_Report.pdf          # Full LaTeX Technical Report
│
├── README.md                         # Project documentation and CLI usage guide
├── requirements.txt                  # Pinned dependencies
├── .gitattributes                    # Git line endings & binary configuration
└── .gitignore                        # Git exclusion rules
```

---

## 3. Dataset Information

### English (Svarah Multi-Speaker Indian English)
- **Source**: Svarah Indian English Corpus.
- **Speakers**: 104 speakers (54 female, 50 male) providing natural acoustic diversity.
- **Duration**: ~5 hours, segmented into 2,752 clean utterances.
- **Split**: 2,476 training utterances / 276 validation utterances (stratified across speakers).
- **Alignment**: Montreal Forced Aligner (MFA) v2.2 with CMUDict + Svarah acoustic model.
- **Speaker Brightness Selection**: Speaker `spk_028` demonstrates high spectral centroid (1624–2223 Hz), yielding crisp consonant bursts (`/s/`, `/t/`, `/k/`) and clear word separation.

### Telugu (IndicTTS Bilingual Corpus)
- **Source**: IndicTTS (IIT Madras / MeitY).
- **Speakers**: 2 native speakers (1 male, 1 female), ~2.5 hours per speaker (~5 hours total).
- **Split**: 4,496 training utterances / 916 validation utterances.
- **Alignment**: Meta MMS (Massively Multilingual Speech) forced aligner trained on Telugu phoneme CTC emissions.
- **Orthography**: Syllabic *akshara* representation maintaining strict 1-to-1 acoustic mapping and natural open-vowel word endings.

---

## 4. Google Drive Model Weights Download

Due to GitHub's 100 MB standard file upload limit, all pre-trained FastSpeech 2 checkpoints and vocoder weights are hosted on Google Drive:

🔗 **[Google Drive Shared Model Folder](https://drive.google.com/drive/folders/1MYS5zcsyehxDN5n4XaKj5jCidpUZ-efw?usp=sharing)**

### Available Files in Google Drive
```text
I:\My Drive\TTS/
├── English/
│   ├── latest_checkpoint.pth         # 418,747,163 bytes (399.35 MB) [Step 100,000 | Epoch 468]
│   ├── best_model.pth                # 418,739,397 bytes (399.34 MB) [Step  61,000 | Epoch 285]
│   ├── English_All_Phases_Logs.zip   #  73,207,678 bytes ( 69.82 MB) [Complete 100k Logs & Events]
│   └── generator_universal.pth.tar   #  55,788,858 bytes ( 53.20 MB)
└── Telugu/
    ├── latest_checkpoint.pth         # 419,457,819 bytes (400.02 MB) [Step 100,000 | Epoch 714]
    ├── best_model.pth                # 419,450,053 bytes (400.02 MB) [Step  74,000 | Epoch 528]
    ├── best_step_64000.pth           # 419,450,053 bytes (400.02 MB) [Step  64,000 | Rank 2]
    ├── best_step_74000.pth           # 419,450,053 bytes (400.02 MB) [Step  74,000 | Rank 1]
    ├── best_step_78000.pth           # 419,450,053 bytes (400.02 MB) [Step  78,000 | Rank 3]
    ├── Telugu_All_Phases_Logs.zip    #  56,073,051 bytes ( 53.48 MB) [Complete 100k Logs & Events]
    ├── Telugu_Phase1_Results.zip     #   3,069,034 bytes (  2.93 MB) [Archived Phase 1 Results]
    └── generator_universal.pth.tar   #  55,788,858 bytes ( 53.20 MB)
```

### File Placement Instructions

After downloading from the shared folder, place each file into its designated directory:

| Downloaded File | Target Repository Path | Role |
| :--- | :--- | :--- |
| `English/latest_checkpoint.pth` | `English/checkpoints/latest_checkpoint.pth` | **Final Completed Checkpoint**: Full model weights, Adam optimizer state, step counter (**Step 100,000 \| Epoch 468 \| Loss: 0.7818 \| Val Loss: 2.6288**). |
| `English/best_model.pth` | `English/checkpoints/best_model.pth` | **Best Inference Model**: Rank 1 overall validation model (**Step 61,000 \| Epoch 285 \| Val Loss: 2.5855**) for high-fidelity speech synthesis. |
| `English/generator_universal.pth.tar` | `English/inference/hifigan/generator_universal.pth.tar` | HiFi-GAN 16 kHz universal neural vocoder weights (also copy to `English/train/hifigan/`). |
| `Telugu/latest_checkpoint.pth` | `Telugu/checkpoints/latest_checkpoint.pth` | **Final Completed Checkpoint**: Full model weights, Adam optimizer state, step counter (**Step 100,000 \| Epoch 714 \| Loss: 0.5841 \| Val Loss: 2.1661**). |
| `Telugu/best_model.pth` | `Telugu/checkpoints/best_model.pth` | **Best Inference Model**: Rank 1 Phase 2 validation model (**Step 74,000 \| Epoch 528 \| Val Loss: 2.1457**) for speech synthesis. |
| `Telugu/generator_universal.pth.tar` | `Telugu/inference/hifigan/generator_universal.pth.tar` | HiFi-GAN 16 kHz universal neural vocoder weights (also copy to `Telugu/train/hifigan/`). |

### Resumable vs. Best Inference Checkpoints
- **`latest_checkpoint.pth`**: Contains the full PyTorch training state (including `model`, `optimizer`, `step`, `epoch`, `best_loss`, and CUDA random state). It is used to seamlessly continue training where the previous session left off.
- **`best_model.pth`**: Represents the model weights that achieved the lowest validation loss across the entire training trajectory. It is specifically optimized for inference and evaluation.

---

## 5. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/savioshaju/TTS
cd TTS

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 6. Preprocessing Instructions

To rerun preprocessing from raw data:

### English (MFA Pipeline)
```bash
cd English/preprocess
python preprocess.py --config ../config.yaml --stage all --jobs 8
```

### Telugu (MMS Pipeline)
```bash
cd Telugu/preprocess
python preprocess.py --config ../config.yaml
```

---

## 7. Training Instructions

The training engine includes memory-leak safeguards (`malloc_trim(0)`, explicit garbage collection, and tensor deallocation). If a checkpoint exists in `checkpoints/`, training **automatically resumes** from that exact step.

### English Training
```bash
python English/train/train.py \
    --config English/config.yaml \
    --data_dir preprocessed_data \
    --output_dir English/output \
    --batch_size 16 \
    --total_step 100000 \
    --resume
```

### Telugu Training
```bash
python Telugu/train/train.py \
    --config Telugu/config.yaml \
    --data_dir preprocessed_data \
    --output_dir Telugu/output \
    --batch_size 16 \
    --total_step 100000 \
    --resume
```

### Dual-GPU Kaggle Training
Open `English/English_FastSpeech2_Kaggle.ipynb` or `Telugu/Telugu_FastSpeech2_Kaggle.ipynb` in Kaggle with accelerator set to **GPU T4 x2**. The notebook automatically configures DataParallel across both GPUs, restores checkpoints, and executes training.

---

## 8. Speech Synthesis


### Custom Text Synthesis (English & Telugu)
To synthesize arbitrary custom sentences from the command line:

#### English Custom Synthesis
```bash
# Synthesize custom English text
python English/synthesize_custom.py --text "Hi, I am Savio." --output custom_synthesize/hi_i_am_savio.wav

# Optional: adjust duration speed (d_control) or speaker
python English/synthesize_custom.py --text "Good morning, welcome to our speech synthesis demonstration." --d_control 1.25 --speaker_id 27 --output custom_synthesize/demo_en.wav
```

#### Telugu Custom Synthesis
```bash
# Synthesize custom Telugu text
python Telugu/synthesize_custom.py --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" --output custom_synthesize/telugu_sample.wav

# Optional: adjust duration speed (d_control)
python Telugu/synthesize_custom.py --text "ఈ రోజు వాతావరణం చాలా బాగుంది." --d_control 1.25 --output custom_synthesize/weather_te.wav
```
---

## 9. Deliverables & Results

### Official Benchmark Utterances

| ID | Language | Sentence / Linguistic Phenomenon | Generated Audio |
| :--- | :--- | :--- | :--- |
| **E1** | English | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.*  | [`Results/English/generated_audios/E1.wav`](Results/English/generated_audios/E1.wav) |
| **E2_a** | English | *The orchestra's conduct during the ceremony was praised by the audience.* | [`Results/English/generated_audios/E2_a.wav`](Results/English/generated_audios/E2_a.wav) |
| **E2_b** | English | *The researchers will conduct an experiment to evaluate the proposed method.* | [`Results/English/generated_audios/E2_b.wav`](Results/English/generated_audios/E2_b.wav) |
| **E3** | English | *The rhyth  m of the orchestra changed unexpectedly.*  | [`Results/English/generated_audios/E3.wav`](Results/English/generated_audios/E3.wav) |
| **E4** | English | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* (Numbers + Code-mixed) | [`Results/English/generated_audios/E4.wav`](Results/English/generated_audios/E4.wav) |
| **T1** | Telugu | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.  | [`Results/Telugu/generated_audios/T1.wav`](Results/Telugu/generated_audios/T1.wav) |
| **T2_a** | Telugu | తాడుపై అతని పట్టు చాలా బలంగా ఉంది.  | [`Results/Telugu/generated_audios/T2_a.wav`](Results/Telugu/generated_audios/T2_a.wav) |
| **T2_b** | Telugu | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. (Adjective/Noun: silk) | [`Results/Telugu/generated_audios/T2_b.wav`](Results/Telugu/generated_audios/T2_b.wav) |
| **T3** | Telugu | పులి అడ  విలో నిశ్శబ్దంగా నడిచింది. | [`Results/Telugu/generated_audios/T3.wav`](Results/Telugu/generated_audios/T3.wav) |
| **T4** | Telugu | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.  | [`Results/Telugu/generated_audios/T4.wav`](Results/Telugu/generated_audios/T4.wav) |

### Variance Adaptor Visualizations
- **Duration Alignment Plots**: Located in `Results/English/variance_adaptor_plots/duration/` and `Results/Telugu/variance_adaptor_plots/duration/`.
- **Pitch ($F_0$) Contours**: Located in `Results/English/variance_adaptor_plots/pitch/` and `Results/Telugu/variance_adaptor_plots/pitch/`.
- **Energy Tracking Contours**: Located in `Results/English/variance_adaptor_plots/energy/` and `Results/Telugu/variance_adaptor_plots/energy/`.

### Training & Validation Loss Trajectories
- **English Loss Curves (100k Steps)**:
  [`Results/English/training_loss.png`](Results/English/training_loss.png) & [`Results/English/validation_loss.png`](Results/English/validation_loss.png)
  - Phase 1 (Steps 100 – 53,700): [`Results/English/phase_1_loss.png`](Results/English/phase_1_loss.png)
  - Phase 2 (Steps 53,100 – 100,000): [`Results/English/phase_2_loss.png`](Results/English/phase_2_loss.png)
- **Telugu Loss Curves (100k Steps)**:
  - Full 100k Trajectory: [`Results/Telugu/training_loss.png`](Results/Telugu/training_loss.png) & [`Results/Telugu/validation_loss.png`](Results/Telugu/validation_loss.png)
  - Phase 1 (Steps 100 – 56,000): [`Results/Telugu/phase_1_loss.png`](Results/Telugu/phase_1_loss.png)
  - Phase 2 (Steps 56,100 – 100,000): [`Results/Telugu/phase_2_loss.png`](Results/Telugu/phase_2_loss.png)

### Formal Technical Report
A comprehensive academic report detailing mathematical formulations, experimental setup, spectrogram analyses, and comparative findings is available at:
[`Report/Technical_Report.pdf`](Report/Technical_Report.pdf).
