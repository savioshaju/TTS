# English FastSpeech 2 Model Checkpoints

Due to GitHub's standard file size limits, large pre-trained model weights (`*.pth`) and HiFi-GAN vocoder weights (`*.pth.tar`) are hosted externally on Google Drive.

---

## Google Drive Download Link

Download the English model weights from the shared Google Drive folder:  
👉 **[Google Drive Shared Model Folder](https://drive.google.com/drive/folders/1MYS5zcsyehxDN5n4XaKj5jCidpUZ-efw?usp=sharing)**

Navigate to the `English/` subfolder in Google Drive:
- `latest_checkpoint.pth` (~399.35 MB / 418,747,163 bytes)
- `best_model.pth` (~399.34 MB / 418,739,397 bytes)
- `best_step_61000.pth` (~399.34 MB / 418,739,397 bytes)
- `best_step_65000.pth` (~399.34 MB / 418,739,397 bytes)
- `best_step_66000.pth` (~399.34 MB / 418,739,397 bytes)
- `generator_universal.pth.tar` (~53.20 MB / 55,788,858 bytes)

---

## File Placement Instructions

After downloading, place the files into this repository as follows:

| Downloaded File | Target Repository Path | Description |
| :--- | :--- | :--- |
| `latest_checkpoint.pth` | `English/checkpoints/latest_checkpoint.pth` | Final converged training checkpoint (**Step 100,000 \| Epoch 468 \| Loss: 0.7818 \| Val Loss: 2.6288**) containing full model weights, Adam optimizer states, and learning rate scheduler. |
| `best_model.pth` | `English/checkpoints/best_model.pth` | **Primary Evaluation Model** (**Step 61,000 \| Epoch 285 \| Val Loss: 2.5855**) selected for high-fidelity speech synthesis. |
| `generator_universal.pth.tar` | `English/inference/hifigan/generator_universal.pth.tar` | HiFi-GAN 16 kHz universal neural vocoder weights (also copy to `English/train/hifigan/`). |

---

## Checkpoint Retention & Evaluation Summary

Training was conducted on Kaggle Dual NVIDIA T4 GPUs using `torch.nn.DataParallel` up to **Step 100,000**. The automated checkpoint retention manager maintained the top validation models plus the latest resumable checkpoint:

| Candidate | Checkpoint File | Training Step | Epoch | Validation Loss | Role |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Primary** | **`best_step_61000.pth` (`best_model.pth`)** | **61,000** | **285** | **2.5855** | Primary evaluation & inference model (crisp formants, optimal prosody) |
| **Secondary** | `best_step_66000.pth` | 66,000 | 309 | 2.5892 | Secondary validation candidate |
| **Tertiary** | `best_step_65000.pth` | 65,000 | 304 | 2.5899 | Tertiary validation candidate |
| **Final** | **`latest_checkpoint.pth`** | **100,000** | **468** | **2.6288** | Fully completed 100k training run (Training Loss: 0.7818) |

---

## CLI Usage Instructions

### 1. Resume Training
```bash
python English/train/train.py --config English/config.yaml --resume
```

### 2. Synthesize with the Best Model
```bash
python English/inference/generate_assignment_audios.py
```
Outputs are automatically written to `Results/English/test_generated_audios/` and `Results/English/generated_audios/`.
