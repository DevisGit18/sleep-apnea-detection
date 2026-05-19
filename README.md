# Sleep Apnea Detection from Physiological Signals

A machine learning pipeline for detecting breathing irregularities during sleep using nasal airflow, thoracic movement, and SpO₂ signals.

## Dataset
Overnight sleep recordings (8 hours) from 5 participants. Each participant folder contains:
- Nasal airflow signal (32 Hz)
- Thoracic movement signal (32 Hz)
- SpO₂ signal (4 Hz)
- Flow events file (annotated breathing irregularities)
- Sleep profile file (sleep stages)
Classes: Hypopnea, Normal, Obstructive Apnea, Other

**Note:** Low precision/recall is expected due to heavy class imbalance
(~91% Normal windows). Class weights were used to partially address this.

