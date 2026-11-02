# Telugu FastSpeech 2 Model Checkpoints

Due to GitHub's standard file size limits, large pre-trained model weights (`*.pth`), HiFi-GAN vocoder weights (`*.pth.tar`), and training archive packages are hosted externally on Google Drive.

---

## Google Drive Download Link

Download the Telugu model weights and full multi-phase logs from the shared Google Drive folder:
👉 **[Google Drive Shared Model Folder](https://drive.google.com/drive/folders/1MYS5zcsyehxDN5n4XaKj5jCidpUZ-efw?usp=sharing)**

Navigate to the `Telugu/` subfolder in Google Drive:
- `best_model.pth` (~400 MB) — Top Phase 2 model (Step 74,000, Val Loss: **2.1457**)
- `latest_checkpoint.pth` (~400 MB) — Terminal 100k training checkpoint (Step 100,000, Epoch 714, Val Loss: **2.1661**)
- `best_step_74000.pth` (~400 MB) — Rank 1 Phase 2 snapshot
- `best_step_64000.pth` (~400 MB) — Rank 2 Phase 2 snapshot (Val Loss: 2.1462)
- `best_step_78000.pth` (~400 MB) — Rank 3 Phase 2 snapshot (Val Loss: 2.1477)
- `Telugu_All_Phases_Logs.zip` (53.48 MB) — Complete Phase 1 and Phase 2 TensorBoard events, PDF execution logs, master combined logs, and metrics
- `Telugu_Phase1_Results.zip` (2.93 MB) — Archived Phase 1 assets and loss plots
- `generator_universal.pth.tar` (~55 MB) — HiFi-GAN universal vocoder weights

---

## File Placement Instructions

After downloading, place the files into this repository as follows:

| Downloaded File | Target Repository Path | Description |
| :--- | :--- | :--- |
| `best_model.pth` | `Telugu/checkpoints/best_model.pth` | Top-performing validation model (Step 74,000, Val Loss: 2.1457) used for speech synthesis. |
| `latest_checkpoint.pth` | `Telugu/checkpoints/latest_checkpoint.pth` | Terminal 100k training checkpoint (Step 100,000) containing model weights, Adam optimizer states, and learning rate scheduler state. |
| `generator_universal.pth.tar` | `Telugu/inference/hifigan/generator_universal.pth.tar` | HiFi-GAN 16 kHz universal neural vocoder weights (also copy to `kaggle_staging/Telugu_FastSpeech2_Kaggle/hifigan/`). |

---

## Checkpoint Details & Convergence Metrics

- **`best_model.pth` (Step 74,000 | Epoch 528 | Val Loss: 2.1457)**:
  Used for **inference and evaluation** (generates natural, accurate Telugu speech).
  ```bash
  python Telugu/generate_telugu_assignment_audios.py
  ```

- **`latest_checkpoint.pth` (Step 100,000 | Epoch 714 | Val Loss: 2.1661 | Train Loss: 0.5841)**:
  Terminal state of the 100,000-step training trajectory.
  - Phase 1 Range: Steps 100 – 56,000 (Best: Step 30,000, Val Loss 2.1309)
  - Phase 2 Range: Steps 56,100 – 100,000 (Best: Step 74,000, Val Loss 2.1457)
