"""
Quick test of AWD-Stacking with minimal data for fast verification.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from preprocessing import DataPreprocessor, get_prediction_horizon_points
from ensemble import SimplifiedAWDStacking
from metrics import rmse, mae, mcc

print("\n" + "="*60)
print("AWD-Stacking Quick Test")
print("="*60)

# Generate minimal synthetic data
np.random.seed(42)
n_samples = 2000

# Simple glucose simulation
t = np.arange(n_samples)
baseline = 120
circadian = 15 * np.sin(2 * np.pi * t / 288)
noise = np.random.normal(0, 10, n_samples)
glucose = baseline + circadian + noise
glucose = np.clip(glucose, 50, 350)

print(f"\nData: {n_samples} samples ({n_samples * 5 / 60:.1f} hours)")
print(f"Range: {glucose.min():.1f} - {glucose.max():.1f} mg/dL")

# Split data
train_data = glucose[:1600]
test_data = glucose[1600:]

# Test for 30-minute prediction
ph_minutes = 30
prediction_horizon = get_prediction_horizon_points(ph_minutes)
history_window = 6

print(f"\nPrediction horizon: {ph_minutes} min ({prediction_horizon} points)")
print(f"History window: {history_window * 5} min ({history_window} points)")

# Preprocess
preprocessor = DataPreprocessor()
train_processed = preprocessor.preprocess(train_data, is_training=True)

X_train, y_train = preprocessor.create_sequences(
    train_processed, history_window, prediction_horizon
)

# Test data
test_processed = preprocessor.interpolate_missing(test_data, is_training=False)
test_processed = preprocessor.kalman_filter.filter_series(test_processed)
test_processed = preprocessor.smoother.smooth(test_processed)
test_processed = preprocessor.normalize(test_processed, fit=False)

X_test, y_test = preprocessor.create_sequences(
    test_processed, history_window, prediction_horizon
)

print(f"Train: {X_train.shape[0]} sequences")
print(f"Test: {X_test.shape[0]} sequences")

# Train ensemble
print("\nTraining ensemble (minimal epochs for speed)...")
ensemble = SimplifiedAWDStacking(
    history_window=history_window,
    prediction_horizon=prediction_horizon,
    n_folds=2,
    random_seed=42
)

ensemble.fit(X_train, y_train, epochs=10, batch_size=32, patience=3, verbose=0)

# Predict
y_pred = ensemble.predict(X_test)

# Denormalize
y_test_denorm = preprocessor.denormalize(y_test)
y_pred_denorm = preprocessor.denormalize(y_pred)

# Final values
y_true_final = y_test_denorm[:, -1]
y_pred_final = y_pred_denorm[:, -1]

# Metrics
print("\n" + "-"*60)
print("Results (minimal training):")
print("-"*60)
print(f"RMSE: {rmse(y_true_final, y_pred_final):.3f} mg/dL")
print(f"MAE:  {mae(y_true_final, y_pred_final):.3f} mg/dL")
print(f"MCC:  {mcc(y_true_final, y_pred_final):.3f}")

print("\n" + "="*60)
print("Quick test completed successfully!")
print("="*60)
print("\nNote: For production results, use:")
print("  - epochs=500, patience=10, n_folds=5")
print("  - Multi-history windows [6, 9, 12, 18]")
print("  - Full AWDStackingEnsemble class")
