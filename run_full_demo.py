"""
Full demonstration of AWD-Stacking with realistic synthetic glucose data.

Since OhioT1DM requires a Data Use Agreement, this script demonstrates
the full pipeline with synthetic data that mimics real CGM patterns.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from preprocessing import DataPreprocessor, get_prediction_horizon_points
from ensemble import SimplifiedAWDStacking
from metrics import compute_all_metrics, print_metrics_summary, rmse, mae, mcc


def generate_realistic_glucose_data(n_days: int = 7, seed: int = 42) -> np.ndarray:
    """
    Generate realistic synthetic glucose data mimicking T1D patterns.

    Features:
    - Baseline glucose with circadian variation
    - Post-meal glucose spikes (3 meals/day)
    - Dawn phenomenon (early morning glucose rise)
    - Random hypoglycemic and hyperglycemic events
    - Realistic noise and variability
    """
    np.random.seed(seed)

    samples_per_day = 288  # 5-min intervals
    n_samples = n_days * samples_per_day
    t = np.arange(n_samples)

    # Baseline glucose (mg/dL)
    baseline = 120

    # Circadian rhythm (lower at night, higher during day)
    circadian = 10 * np.sin(2 * np.pi * (t / samples_per_day - 0.25))

    # Dawn phenomenon (glucose rise in early morning)
    dawn_phase = (t % samples_per_day) / samples_per_day
    dawn = 15 * np.exp(-((dawn_phase - 0.25)**2) / 0.01)

    # Meal responses (breakfast ~7am, lunch ~12pm, dinner ~6pm)
    meals = np.zeros(n_samples)
    for day in range(n_days):
        day_start = day * samples_per_day

        # Breakfast (hour 7, sample 84)
        breakfast_time = day_start + 84 + np.random.randint(-6, 6)
        breakfast_size = 40 + np.random.uniform(-10, 20)

        # Lunch (hour 12, sample 144)
        lunch_time = day_start + 144 + np.random.randint(-6, 6)
        lunch_size = 30 + np.random.uniform(-10, 15)

        # Dinner (hour 18, sample 216)
        dinner_time = day_start + 216 + np.random.randint(-12, 6)
        dinner_size = 45 + np.random.uniform(-10, 25)

        for meal_time, meal_size in [(breakfast_time, breakfast_size),
                                      (lunch_time, lunch_size),
                                      (dinner_time, dinner_size)]:
            if 0 <= meal_time < n_samples:
                # Glucose spike and decay (about 2-3 hours)
                decay_rate = 0.025 + np.random.uniform(0, 0.01)
                for i in range(min(60, n_samples - meal_time)):  # ~5 hour window
                    spike = meal_size * (i/15) * np.exp(-decay_rate * i) if i < 15 else \
                           meal_size * np.exp(-decay_rate * (i-15))
                    meals[meal_time + i] += spike

    # Random glucose events (hypoglycemia/hyperglycemia)
    events = np.zeros(n_samples)
    n_events = n_days // 2
    for _ in range(n_events):
        event_time = np.random.randint(0, n_samples - 48)
        event_type = np.random.choice(['hypo', 'hyper'])
        if event_type == 'hypo':
            magnitude = -30 - np.random.uniform(0, 20)
        else:
            magnitude = 50 + np.random.uniform(0, 30)

        for i in range(48):  # ~4 hour event
            events[event_time + i] += magnitude * np.exp(-i/20)

    # Sensor noise (CGM typically has ±10-15 mg/dL noise)
    noise = np.random.normal(0, 8, n_samples)

    # Combine all components
    glucose = baseline + circadian + dawn + meals + events + noise

    # Clip to realistic range and add occasional missing values
    glucose = np.clip(glucose, 40, 400)

    return glucose


def main():
    print("\n" + "="*70)
    print("AWD-Stacking Blood Glucose Prediction - Full Demonstration")
    print("="*70)
    print("\nNote: OhioT1DM dataset requires a Data Use Agreement.")
    print("This demo uses realistic synthetic glucose data.")
    print("To obtain OhioT1DM: https://ohio.qualtrics.com/jfe/form/SV_02QtWEVm7ARIKIl")

    # Generate data (mimicking OhioT1DM: 8 weeks train, 10 days test)
    print("\n" + "-"*70)
    print("Generating synthetic glucose data...")

    train_data = generate_realistic_glucose_data(n_days=56, seed=42)  # 8 weeks
    test_data = generate_realistic_glucose_data(n_days=10, seed=123)  # 10 days

    print(f"Training samples: {len(train_data)} ({len(train_data)//288} days)")
    print(f"Test samples: {len(test_data)} ({len(test_data)//288} days)")
    print(f"Training data range: {train_data.min():.1f} - {train_data.max():.1f} mg/dL")
    print(f"Training data mean: {train_data.mean():.1f} mg/dL")

    # Results storage
    all_results = {}

    # Test for different prediction horizons
    for ph_minutes in [30, 45, 60]:
        print("\n" + "="*70)
        print(f"PREDICTION HORIZON: {ph_minutes} minutes")
        print("="*70)

        prediction_horizon = get_prediction_horizon_points(ph_minutes)
        history_window = 12  # 60 minutes of history

        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            kalman_process_var=1e-5,
            kalman_measurement_var=1e-2,
            smoothing_alpha=0.1,
            smoothing_beta=0.5
        )

        # Preprocess training data
        print("\nPreprocessing training data...")
        train_processed = preprocessor.preprocess(train_data, is_training=True)

        # Create sequences
        X_train, y_train = preprocessor.create_sequences(
            train_processed, history_window, prediction_horizon
        )
        print(f"Training sequences: X={X_train.shape}, y={y_train.shape}")

        # Preprocess test data (without refitting normalization)
        test_processed = preprocessor.interpolate_missing(test_data, is_training=False)
        test_processed = preprocessor.kalman_filter.filter_series(test_processed)
        test_processed = preprocessor.smoother.smooth(test_processed)
        test_processed = preprocessor.normalize(test_processed, fit=False)

        X_test, y_test = preprocessor.create_sequences(
            test_processed, history_window, prediction_horizon
        )
        print(f"Test sequences: X={X_test.shape}, y={y_test.shape}")

        # Split training into train/val
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        # Train simplified ensemble (faster for demo)
        print("\nTraining AWD-Stacking ensemble...")
        ensemble = SimplifiedAWDStacking(
            history_window=history_window,
            prediction_horizon=prediction_horizon,
            n_folds=3,  # Reduced for faster demo
            random_seed=42
        )

        ensemble.fit(
            X_tr, y_tr,
            epochs=30,  # Reduced for demo (paper uses 500)
            batch_size=32,
            patience=5,
            verbose=1
        )

        # Predict on test set
        print("\nMaking predictions...")
        y_pred = ensemble.predict(X_test)

        # Denormalize
        y_test_denorm = preprocessor.denormalize(y_test)
        y_pred_denorm = preprocessor.denormalize(y_pred)

        # Use last prediction (actual prediction horizon)
        y_true_final = y_test_denorm[:, -1]
        y_pred_final = y_pred_denorm[:, -1]

        # Compute metrics
        metrics = compute_all_metrics(y_true_final, y_pred_final)
        all_results[ph_minutes] = metrics

        print_metrics_summary(metrics, ph_minutes)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'PH (min)':<12} {'RMSE (mg/dL)':<15} {'MAE (mg/dL)':<15} {'MCC':<10}")
    print("-"*55)

    for ph, m in all_results.items():
        print(f"{ph:<12} {m['RMSE']:<15.3f} {m['MAE']:<15.3f} {m['MCC']:<10.3f}")

    print("\n" + "-"*70)
    print("Note: Results with synthetic data. Real OhioT1DM data may differ.")
    print("Paper results (OhioT1DM): RMSE=1.425, MAE=0.721, MCC=0.982 for PH=30min")
    print("-"*70)

    return all_results


if __name__ == '__main__':
    main()
