# Sleep Apnea Detection from Physiological Signals

A machine learning pipeline for detecting breathing irregularities during sleep using nasal airflow, thoracic movement, and SpO₂ signals.

## Dataset
Overnight sleep recordings (8 hours) from 5 participants. Each participant folder contains:
- Nasal airflow signal (32 Hz)
- Thoracic movement signal (32 Hz)
- SpO₂ signal (4 Hz)
- Flow events file (annotated breathing irregularities)
- Sleep profile file (sleep stages)

## Project Structure
```
Project/
├── Data/               # raw participant signal files
├── Visualizations/     # generated PDF plots per participant
├── Dataset/            # processed labeled windows (CSV)
├── scripts/
│   ├── vis.py              # visualization script
│   ├── create_dataset.py   # preprocessing + dataset creation
│   └── train_model.py      # CNN training + LOPO evaluation
├── README.md
└── requirements.txt
```

## Usage

**Step 1 — Generate visualization for a participant:**
```bash
python scripts/vis.py -name Data/AP01
```

**Step 2 — Create the dataset:**
```bash
python scripts/create_dataset.py -in_dir Data -out_dir Dataset
```

**Step 3 — Train and evaluate:**
```bash
python scripts/train_model.py -dataset Dataset/breathing_dataset.csv
```

## Methods

### Preprocessing
- Bandpass filter (0.17–0.4 Hz) applied to all signals to retain breathing frequency range
- Signals split into 30-second windows with 50% overlap
- At 32 Hz: window = 960 samples, step = 480 samples
- SpO₂ upsampled from 120 to 960 samples to match other channels

### Labeling
- Each window labeled based on overlap with annotated events
- If >50% of window overlaps with a breathing event → that event label
- Otherwise → Normal

### Model
- 1D CNN with 3 input channels (flow, thoracic, SpO₂)
- Architecture: 3x Conv1D → BatchNorm → ReLU → MaxPool blocks, followed by fully connected layers
- Class weights used to handle imbalance

### Evaluation
- Leave-One-Participant-Out (LOPO) cross-validation across 5 participants
- Metrics: Accuracy, Precision, Recall, Confusion Matrix

## Results

| Fold | Accuracy | Precision | Recall |
|------|----------|-----------|--------|
| AP01 | 0.6762   | 0.3696    | 0.6229 |
| AP02 | 0.7954   | 0.3899    | 0.5241 |
| AP03 | 0.3626   | 0.2513    | 0.1690 |
| AP04 | 0.8333   | 0.2855    | 0.3033 |
| AP05 | 0.5857   | 0.2909    | 0.2946 |
| **Mean** | **0.6506 ± 0.17** | **0.3175 ± 0.05** | **0.3828 ± 0.17** |

Classes: Hypopnea, Normal, Obstructive Apnea, Other

**Note:** Low precision/recall is expected due to heavy class imbalance
(~91% Normal windows). Class weights were used to partially address this.

## Note on AI Tool Usage-- > i have used AI (claude) to write parts of the code .I am capable of explaining any part if required .