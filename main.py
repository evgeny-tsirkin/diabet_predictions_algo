"""
Main Training and Prediction Pipeline for AWD-Stacking Blood Glucose Prediction

This is the main entry point for training and evaluating the AWD-stacking
ensemble model for blood glucose level prediction.

Usage:
    python main.py --data_path /path/to/data --prediction_horizon 30

The pipeline:
1. Load and preprocess CGM data
2. Create multi-history window sequences
3. Train AWD-stacking ensemble with 5-fold CV
4. Evaluate on test set
5. Generate metrics and visualizations
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from preprocessing import (
    DataPreprocessor,
    DEFAULT_HISTORY_WINDOWS,
    PREDICTION_HORIZONS,
    get_prediction_horizon_points
)
from ensemble import AWDStackingEnsemble, SimplifiedAWDStacking
from metrics import compute_all_metrics, print_metrics_summary, plot_clarke_error_grid


def load_ohio_t1dm_data(patient_id: str,
                        data_dir: str,
                        is_training: bool = True) -> np.ndarray:
    """
    Load OhioT1DM dataset for a specific patient.

    The OhioT1DM dataset contains CGM data recorded every 5 minutes.

    Args:
        patient_id: Patient identifier (e.g., '559', '540')
        data_dir: Directory containing the data files
        is_training: Load training or test data

    Returns:
        Blood glucose values as numpy array
    """
    subset = 'train' if is_training else 'test'
    file_path = os.path.join(data_dir, f'{patient_id}-{subset}.csv')

    if not os.path.exists(file_path):
        # Try XML format (OhioT1DM original format)
        file_path = os.path.join(data_dir, f'{patient_id}-ws-{subset}.xml')
        if os.path.exists(file_path):
            return load_ohio_xml(file_path)
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Try common column names
    for col in ['glucose_value', 'glucose', 'bg', 'value', 'glucose_level']:
        if col in df.columns:
            return df[col].values

    # Use first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols[0]].values

    raise ValueError("Could not find glucose values in data file")


def load_ohio_xml(file_path: str) -> np.ndarray:
    """
    Load OhioT1DM data from XML format.

    Args:
        file_path: Path to XML file

    Returns:
        Blood glucose values as numpy array
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(file_path)
    root = tree.getroot()

    glucose_values = []

    for entry in root.findall('.//glucose_level/event'):
        value = entry.get('value')
        if value:
            glucose_values.append(float(value))

    return np.array(glucose_values)


def generate_synthetic_data(n_samples: int = 10000,
                           seed: int = 42) -> np.ndarray:
    """
    Generate synthetic blood glucose data for testing.

    Simulates realistic CGM readings with:
    - Baseline glucose around 120 mg/dL
    - Daily patterns
    - Meal spikes
    - Random noise

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Synthetic glucose values (mg/dL)
    """
    np.random.seed(seed)

    # Time points (5-minute intervals)
    t = np.arange(n_samples)

    # Baseline
    baseline = 120

    # Circadian rhythm (24-hour cycle, 288 samples = 1 day)
    circadian = 15 * np.sin(2 * np.pi * t / 288)

    # Meal spikes (approximately every 4-6 hours)
    meals = np.zeros(n_samples)
    meal_times = np.random.choice(n_samples, size=n_samples // 72, replace=False)
    for mt in meal_times:
        spike = 50 * np.exp(-np.arange(n_samples - mt) / 36)  # ~3 hour decay
        meals[mt:] += spike[:n_samples - mt]

    # Random noise
    noise = np.random.normal(0, 8, n_samples)

    # Combine components
    glucose = baseline + circadian + meals + noise

    # Clip to realistic range
    glucose = np.clip(glucose, 40, 400)

    return glucose


def prepare_multi_history_data(data: np.ndarray,
                               preprocessor: DataPreprocessor,
                               history_windows: List[int],
                               prediction_horizon: int,
                               is_training: bool = True) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Prepare data with multiple history windows.

    Args:
        data: Raw glucose data
        preprocessor: DataPreprocessor instance
        history_windows: List of history window sizes
        prediction_horizon: Number of future timesteps to predict
        is_training: Whether this is training data

    Returns:
        X_dict: Dictionary mapping history_window -> sequences
        y: Target sequences
    """
    # Preprocess data
    processed = preprocessor.preprocess(data, is_training=is_training)

    # Create sequences with multiple history windows
    X_list, y = preprocessor.create_multi_history_sequences(
        processed, history_windows, prediction_horizon
    )

    # Create dictionary mapping window size to sequences
    X_dict = {hw: X_list[i] for i, hw in enumerate(history_windows)}

    return X_dict, y


def train_and_evaluate(train_data: np.ndarray,
                       test_data: np.ndarray,
                       prediction_horizon_minutes: int = 30,
                       history_windows: List[int] = None,
                       epochs: int = 100,
                       batch_size: int = 32,
                       patience: int = 10,
                       verbose: int = 1) -> Dict:
    """
    Train AWD-stacking model and evaluate on test data.

    Args:
        train_data: Training glucose data (raw, mg/dL)
        test_data: Test glucose data (raw, mg/dL)
        prediction_horizon_minutes: Prediction horizon in minutes (30, 45, or 60)
        history_windows: List of history window sizes (in data points)
        epochs: Maximum training epochs
        batch_size: Training batch size
        patience: Early stopping patience
        verbose: Verbosity level

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if history_windows is None:
        history_windows = DEFAULT_HISTORY_WINDOWS

    prediction_horizon = get_prediction_horizon_points(prediction_horizon_minutes)

    if verbose:
        print(f"\n{'='*60}")
        print(f"AWD-Stacking Blood Glucose Prediction")
        print(f"Prediction Horizon: {prediction_horizon_minutes} minutes ({prediction_horizon} points)")
        print(f"History Windows: {history_windows}")
        print(f"{'='*60}")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        kalman_process_var=1e-5,
        kalman_measurement_var=1e-2,
        smoothing_alpha=0.1,
        smoothing_beta=0.5,
        validation_split=0.2
    )

    # Prepare training data
    if verbose:
        print("\nPreparing training data...")

    X_train_dict, y_train = prepare_multi_history_data(
        train_data, preprocessor, history_windows, prediction_horizon, is_training=True
    )

    # Prepare test data (using same normalization parameters)
    if verbose:
        print("Preparing test data...")

    # For test data, we need to preprocess without refitting
    test_processed = preprocessor.interpolate_missing(test_data, is_training=False)
    test_processed = preprocessor.kalman_filter.filter_series(test_processed)
    test_processed = preprocessor.smoother.smooth(test_processed)
    test_processed = preprocessor.normalize(test_processed, fit=False)

    X_test_list, y_test = preprocessor.create_multi_history_sequences(
        test_processed, history_windows, prediction_horizon
    )
    X_test_dict = {hw: X_test_list[i] for i, hw in enumerate(history_windows)}

    # Initialize and train ensemble
    ensemble = AWDStackingEnsemble(
        history_windows=history_windows,
        prediction_horizon=prediction_horizon,
        n_folds=5,
        alpha=0.5,
        random_seed=42
    )

    if verbose:
        print("\nTraining ensemble model...")

    ensemble.fit(
        X_train_dict, y_train,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose
    )

    # Make predictions
    if verbose:
        print("\nMaking predictions on test set...")

    y_pred = ensemble.predict(X_test_dict)

    # Denormalize predictions and true values
    y_test_denorm = preprocessor.denormalize(y_test)
    y_pred_denorm = preprocessor.denormalize(y_pred)

    # Compute metrics
    if verbose:
        print("\nComputing evaluation metrics...")

    # Take the last predicted value for each sequence (final prediction horizon)
    y_true_final = y_test_denorm[:, -1]
    y_pred_final = y_pred_denorm[:, -1]

    metrics = compute_all_metrics(y_true_final, y_pred_final)

    if verbose:
        print_metrics_summary(metrics, prediction_horizon_minutes)

    return {
        'ensemble': ensemble,
        'preprocessor': preprocessor,
        'predictions': y_pred_denorm,
        'true_values': y_test_denorm,
        'metrics': metrics,
        'prediction_horizon': prediction_horizon_minutes
    }


def run_demo(verbose: int = 1):
    """
    Run a demonstration with synthetic data.

    This is useful for testing the pipeline without the OhioT1DM dataset.
    """
    print("\n" + "="*60)
    print("AWD-Stacking Demo with Synthetic Data")
    print("="*60)

    # Generate synthetic data
    print("\nGenerating synthetic glucose data...")
    np.random.seed(42)

    # 8 weeks of data (5-min intervals) for training
    train_data = generate_synthetic_data(n_samples=16128, seed=42)

    # 10 days of data for testing
    test_data = generate_synthetic_data(n_samples=2880, seed=123)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Training data range: {train_data.min():.1f} - {train_data.max():.1f} mg/dL")

    # Run for different prediction horizons
    results = {}

    for ph in [30, 45, 60]:
        print(f"\n{'='*60}")
        print(f"Training for {ph}-minute prediction horizon")
        print("="*60)

        result = train_and_evaluate(
            train_data, test_data,
            prediction_horizon_minutes=ph,
            epochs=50,  # Reduced for demo
            batch_size=32,
            patience=5,
            verbose=verbose
        )

        results[ph] = result

    # Summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    print(f"\n{'PH (min)':<10} {'RMSE':<10} {'MAE':<10} {'MCC':<10}")
    print("-"*40)

    for ph, result in results.items():
        m = result['metrics']
        print(f"{ph:<10} {m['RMSE']:<10.3f} {m['MAE']:<10.3f} {m['MCC']:<10.3f}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AWD-Stacking Blood Glucose Prediction'
    )
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to data directory (OhioT1DM format)'
    )
    parser.add_argument(
        '--patient_id', type=str, default='559',
        help='Patient ID (e.g., 559, 540)'
    )
    parser.add_argument(
        '--prediction_horizon', type=int, default=30,
        choices=[30, 45, 60],
        help='Prediction horizon in minutes'
    )
    parser.add_argument(
        '--epochs', type=int, default=500,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run demo with synthetic data'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='Verbosity level (0, 1, or 2)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo or args.data_path is None:
        # Run demo mode
        results = run_demo(verbose=args.verbose)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(args.output_dir, f'demo_results_{timestamp}.json')

        # Convert to JSON-serializable format
        json_results = {}
        for ph, result in results.items():
            json_results[str(ph)] = {
                'RMSE': float(result['metrics']['RMSE']),
                'MAE': float(result['metrics']['MAE']),
                'MCC': float(result['metrics']['MCC']),
                'Clarke_EGA': result['metrics']['Clarke_EGA']
            }

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    else:
        # Load real data
        print(f"\nLoading data for patient {args.patient_id}...")
        train_data = load_ohio_t1dm_data(
            args.patient_id, args.data_path, is_training=True
        )
        test_data = load_ohio_t1dm_data(
            args.patient_id, args.data_path, is_training=False
        )

        # Train and evaluate
        result = train_and_evaluate(
            train_data, test_data,
            prediction_horizon_minutes=args.prediction_horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose
        )

        # Generate Clarke Error Grid plot
        plot_path = os.path.join(
            args.output_dir,
            f'clarke_ega_{args.patient_id}_ph{args.prediction_horizon}.png'
        )
        plot_clarke_error_grid(
            result['true_values'][:, -1],
            result['predictions'][:, -1],
            title=f'Clarke EGA - Patient {args.patient_id} (PH={args.prediction_horizon}min)',
            save_path=plot_path
        )
        print(f"\nClarke EGA plot saved to: {plot_path}")

        # Save model
        model_path = os.path.join(
            args.output_dir,
            f'model_{args.patient_id}_ph{args.prediction_horizon}'
        )
        result['ensemble'].save(model_path)
        print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()
