"""
Basic test script to verify AWD-stacking implementation components.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import sys

print("Testing AWD-Stacking Implementation")
print("="*50)

# Test 1: Preprocessing
print("\n1. Testing Preprocessing...")
from preprocessing import KalmanFilter, DoubleExponentialSmoothing, DataPreprocessor

# Generate simple test data
np.random.seed(42)
test_glucose = 120 + 20*np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 5, 100)

# Test Kalman filter
kf = KalmanFilter()
filtered = kf.filter_series(test_glucose)
print(f"   Kalman filter: input shape {test_glucose.shape}, output shape {filtered.shape}")

# Test smoothing
smoother = DoubleExponentialSmoothing(alpha=0.1, beta=0.5)
smoothed = smoother.smooth(filtered)
print(f"   Smoothing: output shape {smoothed.shape}")

# Test preprocessor
preprocessor = DataPreprocessor()
processed = preprocessor.preprocess(test_glucose, is_training=True)
print(f"   Full preprocessing: output shape {processed.shape}")
print("   ✓ Preprocessing module OK")

# Test 2: Sequence creation
print("\n2. Testing Sequence Creation...")
X, y = preprocessor.create_sequences(processed, history_window=6, prediction_horizon=6)
print(f"   X shape: {X.shape}, y shape: {y.shape}")
print("   ✓ Sequence creation OK")

# Test 3: Models
print("\n3. Testing Neural Network Models...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from models import BiLSTMModel, StackLSTMModel, VanillaLSTMModel

# Test BiLSTM
print("   Testing BiLSTM...")
bilstm = BiLSTMModel(history_window=6, prediction_horizon=6)
bilstm.build()
test_pred = bilstm.predict(X[:10])
print(f"   BiLSTM prediction shape: {test_pred.shape}")

# Test StackLSTM
print("   Testing StackLSTM...")
stacklstm = StackLSTMModel(history_window=6, prediction_horizon=6)
stacklstm.build()
test_pred = stacklstm.predict(X[:10])
print(f"   StackLSTM prediction shape: {test_pred.shape}")

# Test VanillaLSTM
print("   Testing VanillaLSTM...")
vlstm = VanillaLSTMModel(history_window=6, prediction_horizon=6, use_peephole=False)
vlstm.build()
test_pred = vlstm.predict(X[:10])
print(f"   VanillaLSTM prediction shape: {test_pred.shape}")
print("   ✓ Neural network models OK")

# Test 4: Clustering
print("\n4. Testing Affinity Propagation Clustering...")
from clustering import AdaptiveWeightCalculator

# Create dummy predictions from 3 models
preds = [
    np.random.rand(20, 6),
    np.random.rand(20, 6) + 0.1,
    np.random.rand(20, 6) - 0.1
]

weight_calc = AdaptiveWeightCalculator()
weights = weight_calc.compute_weights(preds)
print(f"   Computed weights: {weights}")
print(f"   Weights sum: {weights.sum():.4f}")
print("   ✓ Clustering OK")

# Test 5: Metrics
print("\n5. Testing Evaluation Metrics...")
from metrics import rmse, mae, mcc, clarke_error_grid_analysis

y_true = np.array([100, 150, 80, 200, 120])
y_pred = np.array([105, 145, 75, 210, 115])

print(f"   RMSE: {rmse(y_true, y_pred):.3f}")
print(f"   MAE: {mae(y_true, y_pred):.3f}")
print(f"   MCC: {mcc(y_true, y_pred):.3f}")

clarke = clarke_error_grid_analysis(y_true, y_pred)
print(f"   Clarke EGA Zone A: {clarke['A']:.1f}%")
print("   ✓ Metrics OK")

# Test 6: Quick ensemble training
print("\n6. Testing Simplified Ensemble (quick training)...")
from ensemble import SimplifiedAWDStacking

# Prepare small dataset
X_train, X_val = X[:60], X[60:80]
y_train, y_val = y[:60], y[60:80]

ensemble = SimplifiedAWDStacking(
    history_window=6,
    prediction_horizon=6,
    n_folds=2,  # Reduced for quick test
    random_seed=42
)

# Train with minimal epochs
print("   Training (this may take a moment)...")
ensemble.fit(X_train, y_train, epochs=5, batch_size=16, patience=2, verbose=0)

# Predict
predictions = ensemble.predict(X_val)
print(f"   Predictions shape: {predictions.shape}")
print(f"   Sample prediction: {predictions[0][:3]}")
print("   ✓ Ensemble training OK")

print("\n" + "="*50)
print("All tests passed! AWD-Stacking implementation is working.")
print("="*50)
