# Sleep Apnea Detection from Physiological Signals

A 1D CNN pipeline for detecting breathing irregularities during sleep using nasal airflow, thoracic movement, and SpO₂ signals. Evaluated using Leave-One-Participant-Out (LOPO) cross-validation to simulate real-world generalization across unseen subjects.

---

## Problem

Sleep apnea is a common but underdiagnosed condition where breathing repeatedly stops during sleep. Manual scoring of overnight polysomnography recordings is time-consuming and requires specialist expertise. This project builds an automated classifier that can distinguish apnea events from normal breathing using non-invasive physiological signals.

---

## Dataset

Overnight sleep recordings (~8 hours) from 5 participants. Each participant folder contains:

| File | Signal | Sampling Rate |
|---|---|---|
| Nasal airflow | Breathing effort at the nose | 32 Hz |
| Thoracic movement | Chest wall motion | 32 Hz |
| SpO₂ | Blood oxygen saturation | 4 Hz |
| Flow events | Annotated breathing irregularities | - |
| Sleep profile | Sleep stage labels | - |

**Classes:** Hypopnea, Normal, Obstructive Apnea, Other

**Class distribution:** ~91% Normal windows — heavily imbalanced. Class weights are applied during training to partially address this.

---

## Pipeline

```
Raw signals (nasal flow, thoracic, SpO₂)
        |
        v
  create_dataset.py
  - Sliding window segmentation (960 samples @ 32 Hz = 30s windows)
  - SpO₂ upsampled 8x (4 Hz -> 32 Hz) to match other channels
  - Label assignment from annotation files
  - Output: breathing_dataset.csv
        |
        v
  train_model.py
  - LOPO cross-validation (leave one participant out per fold)
  - 1D CNN with 3 conv blocks + classifier head
  - Class-weighted CrossEntropyLoss
  - Adam optimizer with StepLR scheduler
        |
        v
  vis.py
  - Per-participant signal visualizations
  - Confusion matrices per fold
```

---

## Model Architecture

**Input:** `(batch, 3, 960)` — 3 channels (flow, thoracic, SpO₂ upsampled), 960 time steps

```
Conv1D(3->32,  k=7) + BN + ReLU + MaxPool(4)
Conv1D(32->64, k=5) + BN + ReLU + MaxPool(4)
Conv1D(64->128,k=3) + BN + ReLU + MaxPool(4)
Flatten
Linear(128*15 -> 256) + ReLU + Dropout(0.4)
Linear(256 -> 4)
```

---

## Evaluation — LOPO Cross-Validation

Leave-One-Participant-Out: each fold trains on 4 participants and tests on the held-out participant. This repeats for all 5 participants. This is a stricter evaluation than a random train/test split — the model never sees any data from the test subject during training, which better reflects real-world deployment.

**Note:** Low precision/recall on minority classes is expected given the 91% Normal class imbalance. Class weights partially mitigate this but do not fully resolve it with only 5 participants.

---

## Repository Structure

```
sleep-apnea-detection/
├── create_dataset.py          # Signal loading, windowing, dataset construction
├── train_model.py             # 1D CNN definition, LOPO training and evaluation
├── vis.py                     # Signal and result visualizations
├── requirements.txt           # Python dependencies
├── AP01_visualization-1.pdf   # Per-participant signal visualizations
├── AP02_visualization.pdf
├── AP03_visualization.pdf
├── AP04_visualization.pdf
├── AP05_visualization.pdf
└── med_ai(1).ipynb            # Exploratory analysis notebook
```

---

## Setup and Usage

```bash
git clone https://github.com/DevisGit18/sleep-apnea-detection.git
cd sleep-apnea-detection
pip install -r requirements.txt
```

**Build dataset:**
```bash
python create_dataset.py
```

**Train and evaluate:**
```bash
python train_model.py -dataset Dataset/breathing_dataset.csv
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | 1D CNN (PyTorch) |
| Evaluation | LOPO cross-validation |
| Signal processing | NumPy, SciPy |
| Class imbalance handling | sklearn compute_class_weight |
| Visualisation | Matplotlib |
| Dataset | Custom overnight PSG recordings (5 participants) |
