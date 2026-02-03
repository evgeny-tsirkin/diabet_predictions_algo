# AWD-Stacking: Blood Glucose Level Prediction

Implementation of the AWD-stacking (Adaptive Weighted Deep Stacking) ensemble learning algorithm for predicting blood glucose levels in Type 1 Diabetes patients, as described in:

> Yang, H., Chen, Z., Huang, J., & Li, S. (2024). AWD-stacking: An enhanced ensemble learning model for predicting glucose levels. *PLoS ONE*, 19(2), e0291594.

## Overview

The AWD-stacking model predicts future blood glucose levels (30, 45, or 60 minutes ahead) using continuous glucose monitoring (CGM) data. The algorithm combines:

1. **Data Preprocessing**:
   - Kalman filtering for CGM sensor error correction
   - Double exponential smoothing for noise reduction
   - Min-Max normalization

2. **Base Estimators** (LSTM variants):
   - BiLSTM (Bidirectional LSTM)
   - StackLSTM (Stacked 3-layer LSTM)
   - VanillaLSTM (LSTM with peephole connections)

3. **Ensemble Learning**:
   - Stacking with 5-fold cross-validation
   - Improved affinity propagation clustering for adaptive weighting
   - Linear regression meta-model
   - Multi-history window technique (30, 45, 60, 90 minutes)

## Project Structure

```
predictions_algo/
├── preprocessing.py    # Kalman filter, smoothing, normalization
├── models.py          # BiLSTM, StackLSTM, VanillaLSTM implementations
├── clustering.py      # Weighted affinity propagation clustering
├── ensemble.py        # AWD-stacking ensemble model
├── metrics.py         # RMSE, MAE, MCC, Clarke Error Grid
├── main.py           # Training and evaluation pipeline
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

```bash
# Clone or download the code
cd predictions_algo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Demo Mode (Synthetic Data)

Run a demonstration with synthetic glucose data:

```bash
python main.py --demo
```

### With OhioT1DM Dataset

1. Obtain the OhioT1DM dataset from: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html

2. Run training:

```bash
python main.py \
    --data_path /path/to/ohio_data \
    --patient_id 559 \
    --prediction_horizon 30 \
    --epochs 500
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | None | Path to OhioT1DM data directory |
| `--patient_id` | 559 | Patient ID (559, 563, 570, etc.) |
| `--prediction_horizon` | 30 | Prediction horizon in minutes (30, 45, 60) |
| `--epochs` | 500 | Maximum training epochs |
| `--batch_size` | 32 | Training batch size |
| `--demo` | False | Run demo with synthetic data |
| `--output_dir` | ./results | Output directory for results |
| `--verbose` | 1 | Verbosity level (0, 1, 2) |

## Model Architecture

### Base Models

1. **BiLSTM**: Bidirectional LSTM with 128 units, processes sequences in both forward and backward directions.

2. **StackLSTM**: 3-layer stacked LSTM with 128→64→32 units, with dropout (0.2) between layers.

3. **VanillaLSTM**: Standard LSTM with 128 units and peephole connections for better long-term dependency capture.

### Ensemble

The stacking ensemble:
1. Trains each base model using 5-fold cross-validation
2. Uses out-of-fold predictions as features for meta-model
3. Applies adaptive weighting via improved affinity propagation clustering
4. Combines predictions using linear regression meta-model

## Evaluation Metrics

- **RMSE** (Root Mean Square Error): Primary regression metric
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **MCC** (Matthews Correlation Coefficient): Classification performance for adverse events
- **Clarke EGA** (Error Grid Analysis): Clinical accuracy assessment

## Results (from paper)

| PH (min) | RMSE (mg/dL) | MAE (mg/dL) | MCC |
|----------|--------------|-------------|-----|
| 30 | 1.425 | 0.721 | 0.982 |
| 45 | 3.212 | 1.605 | 0.950 |
| 60 | 6.346 | 3.232 | 0.930 |

## Citation

If you use this code, please cite:

```bibtex
@article{yang2024awd,
  title={AWD-stacking: An enhanced ensemble learning model for predicting glucose levels},
  author={Yang, HuaZhong and Chen, Zhongju and Huang, Jinfan and Li, Suruo},
  journal={PLoS ONE},
  volume={19},
  number={2},
  pages={e0291594},
  year={2024},
  publisher={Public Library of Science}
}
```

## License

This implementation is provided for research and educational purposes.
