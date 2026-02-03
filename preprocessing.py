"""
Data Preprocessing Module for AWD-stacking Blood Glucose Prediction

This module implements:
1. Linear interpolation (training) and extrapolation (test)
2. Kalman filtering for CGM sensor error correction
3. Double exponential smoothing for data smoothing
4. Min-Max normalization
5. Multi-history window data preparation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


class KalmanFilter:
    """
    Kalman filter for correcting CGM sensor readings.

    The Kalman filter addresses discrepancies between interstitial fluid
    glucose levels (measured by CGM) and actual blood glucose levels.
    """

    def __init__(self,
                 process_variance: float = 1e-5,
                 measurement_variance: float = 1e-2,
                 initial_estimate: float = 0.0,
                 initial_error: float = 1.0):
        """
        Initialize Kalman Filter.

        Args:
            process_variance: Q - process noise covariance
            measurement_variance: R - observation noise covariance
            initial_estimate: Initial state estimate
            initial_error: Initial error covariance
        """
        self.Q = process_variance  # Process noise covariance
        self.R = measurement_variance  # Measurement noise covariance
        self.x_hat = initial_estimate  # State estimate
        self.P = initial_error  # Error covariance

    def update(self, measurement: float) -> float:
        """
        Update the Kalman filter with a new measurement.

        Implements the two-stage Kalman filter:
        1. Prediction (time update)
        2. Correction (measurement update)

        Args:
            measurement: New observation (CGM reading)

        Returns:
            Corrected glucose value
        """
        # Prediction (Time Update)
        # x_k = A * x_{k-1} + B * u_{k-1} + w_{k-1}
        # For univariate case with A=1, B=0: x_k_prior = x_{k-1}
        x_prior = self.x_hat

        # P_k = A * P_{k-1} * A^T + Q
        P_prior = self.P + self.Q

        # Correction (Measurement Update)
        # K_k = P_k * H^T * (H * P_k * H^T + R)^{-1}
        # For univariate case with H=1:
        K = P_prior / (P_prior + self.R)  # Kalman gain

        # x_hat_k = x_k_prior + K * (z_k - H * x_k_prior)
        self.x_hat = x_prior + K * (measurement - x_prior)

        # P_k = (I - K * H) * P_k_prior
        self.P = (1 - K) * P_prior

        return self.x_hat

    def filter_series(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Kalman filter to entire time series.

        Args:
            data: Input time series

        Returns:
            Filtered time series
        """
        # Reset state for new series
        self.x_hat = data[0] if len(data) > 0 else 0.0
        self.P = 1.0

        filtered = np.zeros_like(data)
        for i, measurement in enumerate(data):
            filtered[i] = self.update(measurement)

        return filtered


class DoubleExponentialSmoothing:
    """
    Double Exponential Smoothing (Holt's method) for time series data.

    Captures both level and trend components in the data to eliminate
    noise and outliers while preserving the underlying pattern.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.5):
        """
        Initialize Double Exponential Smoothing.

        Args:
            alpha: Level component smoothing coefficient (0 < alpha < 1)
            beta: Trend component smoothing coefficient (0 < beta < 1)
        """
        self.alpha = alpha
        self.beta = beta

    def smooth(self, data: np.ndarray) -> np.ndarray:
        """
        Apply double exponential smoothing to time series.

        Level equation: L_t = α * Y_t + (1 - α) * (L_{t-1} + T_{t-1})
        Trend equation: T_t = β * (L_t - L_{t-1}) + (1 - β) * T_{t-1}
        Smoothed value: Y_t = L_t + T_t

        Args:
            data: Input time series

        Returns:
            Smoothed time series
        """
        n = len(data)
        if n == 0:
            return data

        # Initialize level and trend
        L = np.zeros(n)
        T = np.zeros(n)
        smoothed = np.zeros(n)

        # Initial values
        L[0] = data[0]
        T[0] = data[1] - data[0] if n > 1 else 0
        smoothed[0] = L[0] + T[0]

        for t in range(1, n):
            # Level component smoothing (Eq. 7)
            L[t] = self.alpha * data[t] + (1 - self.alpha) * (L[t-1] + T[t-1])

            # Trend component smoothing (Eq. 8)
            T[t] = self.beta * (L[t] - L[t-1]) + (1 - self.beta) * T[t-1]

            # Smoothed value (Eq. 9)
            smoothed[t] = L[t] + T[t]

        return smoothed


class DataPreprocessor:
    """
    Complete data preprocessing pipeline for blood glucose prediction.

    Implements the 6-step preprocessing sequence from the paper:
    1. Collect dataset
    2. Linear interpolation (train) / extrapolation (test)
    3. Kalman filtering
    4. Double exponential smoothing
    5. Min-Max normalization
    6. Train/validation split (80/20)
    """

    def __init__(self,
                 kalman_process_var: float = 1e-5,
                 kalman_measurement_var: float = 1e-2,
                 smoothing_alpha: float = 0.1,
                 smoothing_beta: float = 0.5,
                 validation_split: float = 0.2):
        """
        Initialize preprocessor with all components.

        Args:
            kalman_process_var: Kalman filter process variance
            kalman_measurement_var: Kalman filter measurement variance
            smoothing_alpha: Double exponential smoothing alpha
            smoothing_beta: Double exponential smoothing beta
            validation_split: Fraction of training data for validation
        """
        self.kalman_filter = KalmanFilter(
            process_variance=kalman_process_var,
            measurement_variance=kalman_measurement_var
        )
        self.smoother = DoubleExponentialSmoothing(
            alpha=smoothing_alpha,
            beta=smoothing_beta
        )
        self.validation_split = validation_split

        # Normalization parameters (fitted on training data)
        self.min_val = None
        self.max_val = None

    def interpolate_missing(self, data: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Handle missing values using linear interpolation (train) or extrapolation (test).

        Args:
            data: Input data with potential NaN values
            is_training: If True, use interpolation; else use extrapolation

        Returns:
            Data with missing values filled
        """
        df = pd.Series(data)

        if is_training:
            # Linear interpolation for training data
            df = df.interpolate(method='linear', limit_direction='both')
        else:
            # Linear extrapolation for test data (forward fill then backward fill)
            df = df.ffill().bfill()

        return df.values

    def normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Apply Min-Max normalization.

        Args:
            data: Input data
            fit: If True, compute and store min/max from this data

        Returns:
            Normalized data in range [0, 1]
        """
        if fit:
            self.min_val = np.min(data)
            self.max_val = np.max(data)

        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer not fitted. Call with fit=True first.")

        return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse Min-Max normalization.

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer not fitted.")

        return data * (self.max_val - self.min_val) + self.min_val

    def preprocess(self,
                   data: np.ndarray,
                   is_training: bool = True) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Args:
            data: Raw glucose data
            is_training: Whether this is training or test data

        Returns:
            Preprocessed data
        """
        # Step 2: Handle missing values
        data = self.interpolate_missing(data, is_training)

        # Step 3: Kalman filtering
        data = self.kalman_filter.filter_series(data)

        # Step 4: Double exponential smoothing
        data = self.smoother.smooth(data)

        # Step 5: Normalization
        data = self.normalize(data, fit=is_training)

        return data

    def create_sequences(self,
                        data: np.ndarray,
                        history_window: int,
                        prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time series to supervised learning format using sliding windows.

        Args:
            data: Preprocessed time series
            history_window: Number of past observations (input)
            prediction_horizon: Number of future observations (output)

        Returns:
            X: Input sequences of shape (n_samples, history_window, 1)
            y: Target sequences of shape (n_samples, prediction_horizon)
        """
        X, y = [], []

        for i in range(len(data) - history_window - prediction_horizon + 1):
            X.append(data[i:i + history_window])
            y.append(data[i + history_window:i + history_window + prediction_horizon])

        X = np.array(X).reshape(-1, history_window, 1)
        y = np.array(y)

        return X, y

    def create_multi_history_sequences(self,
                                       data: np.ndarray,
                                       history_windows: List[int],
                                       prediction_horizon: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Create sequences with multiple history window sizes.

        The paper uses history windows of 6, 9, 12, and 18 data points
        (corresponding to 30, 45, 60, and 90 minutes of history).

        Args:
            data: Preprocessed time series
            history_windows: List of history window sizes [6, 9, 12, 18]
            prediction_horizon: Number of future observations

        Returns:
            X_list: List of input arrays for each history window
            y: Target sequences (same for all windows)
        """
        max_history = max(history_windows)

        # Use the maximum history window to determine sample count
        X_ref, y = self.create_sequences(data, max_history, prediction_horizon)

        X_list = []
        for hw in history_windows:
            # Extract the last 'hw' timesteps from each sequence
            X_hw = X_ref[:, -hw:, :]
            X_list.append(X_hw)

        return X_list, y

    def split_train_val(self,
                        X: np.ndarray,
                        y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.

        Args:
            X: Input sequences
            y: Target sequences

        Returns:
            X_train, X_val, y_train, y_val
        """
        split_idx = int(len(X) * (1 - self.validation_split))

        X_train = X[:split_idx]
        X_val = X[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]

        return X_train, X_val, y_train, y_val


def get_history_window_minutes(minutes: int) -> int:
    """
    Convert history window in minutes to number of data points.
    CGM data is recorded every 5 minutes.

    Args:
        minutes: History window in minutes

    Returns:
        Number of data points
    """
    return minutes // 5


def get_prediction_horizon_points(minutes: int) -> int:
    """
    Convert prediction horizon in minutes to number of data points.

    Args:
        minutes: Prediction horizon in minutes (30, 45, or 60)

    Returns:
        Number of data points (6, 9, or 12)
    """
    return minutes // 5


# Default configuration from the paper
DEFAULT_HISTORY_WINDOWS = [
    get_history_window_minutes(30),   # 6 data points
    get_history_window_minutes(45),   # 9 data points
    get_history_window_minutes(60),   # 12 data points
    get_history_window_minutes(90),   # 18 data points
]

PREDICTION_HORIZONS = {
    30: get_prediction_horizon_points(30),  # 6 data points
    45: get_prediction_horizon_points(45),  # 9 data points
    60: get_prediction_horizon_points(60),  # 12 data points
}
