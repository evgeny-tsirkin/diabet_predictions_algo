"""
Evaluation Metrics for Blood Glucose Prediction

This module implements the evaluation metrics used in the paper:
1. RMSE (Root Mean Square Error)
2. MAE (Mean Absolute Error)
3. MCC (Matthews Correlation Coefficient)
4. Clarke Error Grid Analysis (EGA)

Metrics are designed for blood glucose prediction where:
- Low: BGL < 70 mg/dL (hypoglycemia)
- Normal: 70 mg/dL ≤ BGL < 126 mg/dL
- High: BGL ≥ 126 mg/dL (hyperglycemia)
"""

import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    RMSE = sqrt(Σ(y_i - ŷ_i)² / n)  (Eq. 20)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    MAE = (1/n) * Σ|y_i - ŷ_i|  (Eq. 21)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def classify_glucose_level(value: float) -> int:
    """
    Classify blood glucose level.

    Categories (per International Diabetes Federation):
    - Low (hypoglycemia): BGL < 70 mg/dL -> returns 0
    - Normal: 70 ≤ BGL < 126 mg/dL -> returns 1
    - High (hyperglycemia): BGL ≥ 126 mg/dL -> returns 2

    Args:
        value: Blood glucose value in mg/dL

    Returns:
        Class label (0=low, 1=normal, 2=high)
    """
    if value < 70:
        return 0  # Hypoglycemia (adverse)
    elif value < 126:
        return 1  # Normal
    else:
        return 2  # Hyperglycemia (adverse)


def is_adverse_event(value: float) -> bool:
    """
    Check if glucose value is an adverse event (hypo or hyper).

    Args:
        value: Blood glucose value in mg/dL

    Returns:
        True if adverse event (hypo or hyper), False if normal
    """
    return value < 70 or value >= 126


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Compute confusion matrix for adverse vs normal glucose events.

    True Positive (TP): Predicted adverse, actual adverse
    True Negative (TN): Predicted normal, actual normal
    False Positive (FP): Predicted adverse, actual normal
    False Negative (FN): Predicted normal, actual adverse

    Args:
        y_true: True glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Convert to binary adverse/normal
    true_adverse = np.array([is_adverse_event(v) for v in y_true])
    pred_adverse = np.array([is_adverse_event(v) for v in y_pred])

    TP = np.sum(pred_adverse & true_adverse)
    TN = np.sum(~pred_adverse & ~true_adverse)
    FP = np.sum(pred_adverse & ~true_adverse)
    FN = np.sum(~pred_adverse & true_adverse)

    return {'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)}


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Matthews Correlation Coefficient.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))  (Eq. 22)

    MCC ranges from -1 to 1:
    - 1: Perfect prediction
    - 0: Random prediction
    - -1: Inverse prediction

    Args:
        y_true: True glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        MCC value
    """
    cm = confusion_matrix_binary(y_true, y_pred)
    TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']

    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def clarke_error_grid_zone(y_true: float, y_pred: float) -> str:
    """
    Determine Clarke Error Grid zone for a single prediction.

    Zone definitions (Table 4 in paper):
    A: Clinically accurate (error within ±20% or both < 70 mg/dL)
    B: Benign errors (would not lead to inappropriate treatment)
    C: Overcorrection errors (could lead to unnecessary treatment)
    D: Dangerous failure to detect (failure to detect hypo/hyperglycemia)
    E: Erroneous treatment (opposite treatment would be given)

    Args:
        y_true: True glucose value (mg/dL)
        y_pred: Predicted glucose value (mg/dL)

    Returns:
        Zone letter ('A', 'B', 'C', 'D', or 'E')
    """
    # Handle edge cases
    if y_true <= 0 or y_pred <= 0:
        return 'E'

    # Zone A: Within ±20% or both in hypoglycemic range
    if (y_true < 70 and y_pred < 70):
        return 'A'

    if y_true >= 70:
        # Relative error within ±20%
        if abs(y_pred - y_true) / y_true <= 0.2:
            return 'A'
    else:
        # Absolute error for hypoglycemic range
        if abs(y_pred - y_true) <= 20:
            return 'A'

    # Zone E: Opposite extremes (most dangerous)
    if (y_true >= 180 and y_pred <= 70) or (y_true <= 70 and y_pred >= 180):
        return 'E'

    # Zone D: Dangerous failure to detect
    if (y_true < 70 and y_pred >= 70 and y_pred < 180):
        # Failed to detect hypoglycemia
        return 'D'
    if (y_true >= 240 and y_pred < 180):
        # Failed to detect severe hyperglycemia
        return 'D'

    # Zone C: Overcorrection errors
    if (y_true >= 70 and y_true <= 180 and y_pred > 180):
        # Would overcorrect for hyperglycemia
        if y_true < 70 + (y_pred - 70) * 0.5:
            return 'C'
    if (y_true >= 70 and y_true <= 180 and y_pred < 70):
        # Would overcorrect for hypoglycemia
        return 'C'

    # Zone B: Benign errors (everything else)
    return 'B'


def clarke_error_grid_analysis(y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    Perform Clarke Error Grid Analysis.

    The EGA evaluates clinical accuracy by categorizing prediction errors
    into five zones (A-E) based on their clinical implications.

    Args:
        y_true: True glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        Dictionary with percentage of predictions in each zone
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    n_samples = len(y_true)
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for true_val, pred_val in zip(y_true, y_pred):
        zone = clarke_error_grid_zone(true_val, pred_val)
        zones[zone] += 1

    # Convert to percentages
    for zone in zones:
        zones[zone] = (zones[zone] / n_samples) * 100

    return zones


def plot_clarke_error_grid(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           title: str = "Clarke Error Grid Analysis",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Clarke Error Grid with data points.

    Args:
        y_true: True glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        title: Plot title
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib Figure object
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot grid boundaries
    ax.plot([0, 400], [0, 400], 'k-', linewidth=0.5)  # Diagonal

    # Zone A boundaries (±20%)
    ax.plot([0, 70], [0, 70], 'k-', linewidth=1)
    ax.plot([70, 400], [70*0.8, 400*0.8], 'k-', linewidth=1)
    ax.plot([70, 400], [70*1.2, 400*1.2], 'k-', linewidth=1)

    # Zone B boundaries
    ax.plot([0, 70], [70, 70], 'k--', linewidth=0.5)
    ax.plot([70, 70], [0, 70], 'k--', linewidth=0.5)
    ax.plot([70, 180], [180, 180], 'k--', linewidth=0.5)
    ax.plot([180, 180], [70, 180], 'k--', linewidth=0.5)

    # Zone labels
    ax.text(30, 30, 'A', fontsize=14, fontweight='bold')
    ax.text(200, 250, 'A', fontsize=14, fontweight='bold')
    ax.text(100, 50, 'B', fontsize=14)
    ax.text(50, 100, 'B', fontsize=14)
    ax.text(300, 200, 'B', fontsize=14)
    ax.text(200, 300, 'B', fontsize=14)
    ax.text(80, 300, 'C', fontsize=14)
    ax.text(300, 80, 'C', fontsize=14)
    ax.text(150, 350, 'D', fontsize=14)
    ax.text(350, 150, 'D', fontsize=14)
    ax.text(50, 350, 'E', fontsize=14)
    ax.text(350, 50, 'E', fontsize=14)

    # Plot data points with color coding
    zones = {}
    for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        zone = clarke_error_grid_zone(true_val, pred_val)
        if zone not in zones:
            zones[zone] = {'x': [], 'y': []}
        zones[zone]['x'].append(true_val)
        zones[zone]['y'].append(pred_val)

    colors = {'A': 'green', 'B': 'blue', 'C': 'yellow', 'D': 'orange', 'E': 'red'}
    for zone, points in zones.items():
        ax.scatter(points['x'], points['y'], c=colors[zone], alpha=0.5,
                  label=f'Zone {zone}', s=10)

    ax.set_xlabel('Reference values of Blood Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicting Glucose Levels (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def error_percentage_analysis(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict[str, float]:
    """
    Analyze prediction errors by percentage ranges.

    Categories:
    - Within ±10%: Excellent
    - Within ±20%: Good
    - Within ±30%: Acceptable
    - Beyond ±30%: Poor

    Args:
        y_true: True glucose values
        y_pred: Predicted glucose values

    Returns:
        Dictionary with percentage of predictions in each error range
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Avoid division by zero
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)

    errors = np.abs(y_pred - y_true) / y_true_safe * 100

    results = {
        'within_10_percent': np.mean(errors <= 10) * 100,
        'within_20_percent': np.mean(errors <= 20) * 100,
        'within_30_percent': np.mean(errors <= 30) * 100,
        'beyond_30_percent': np.mean(errors > 30) * 100,
        'mean_error_percent': np.mean(errors),
        'median_error_percent': np.median(errors)
    }

    return results


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        include_clarke: bool = True) -> Dict[str, any]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: True glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        include_clarke: Whether to include Clarke EGA

    Returns:
        Dictionary with all computed metrics
    """
    results = {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MCC': mcc(y_true, y_pred),
    }

    if include_clarke:
        results['Clarke_EGA'] = clarke_error_grid_analysis(y_true, y_pred)

    results['Error_Analysis'] = error_percentage_analysis(y_true, y_pred)
    results['Confusion_Matrix'] = confusion_matrix_binary(y_true, y_pred)

    return results


def print_metrics_summary(metrics: Dict[str, any], prediction_horizon: int = 30):
    """
    Print a formatted summary of evaluation metrics.

    Args:
        metrics: Dictionary of computed metrics
        prediction_horizon: Prediction horizon in minutes
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Results (PH = {prediction_horizon} minutes)")
    print(f"{'='*50}")

    print(f"\nRegression Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.3f} mg/dL")
    print(f"  MAE:  {metrics['MAE']:.3f} mg/dL")
    print(f"  MCC:  {metrics['MCC']:.3f}")

    if 'Clarke_EGA' in metrics:
        print(f"\nClarke Error Grid Analysis:")
        for zone, pct in metrics['Clarke_EGA'].items():
            status = "✓" if zone in ['A', 'B'] else "✗"
            print(f"  Zone {zone}: {pct:.2f}% {status}")

    if 'Error_Analysis' in metrics:
        print(f"\nError Distribution:")
        ea = metrics['Error_Analysis']
        print(f"  Within ±10%: {ea['within_10_percent']:.2f}%")
        print(f"  Within ±20%: {ea['within_20_percent']:.2f}%")
        print(f"  Mean Error:  {ea['mean_error_percent']:.2f}%")

    print(f"{'='*50}\n")
