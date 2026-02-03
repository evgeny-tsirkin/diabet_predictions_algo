"""
AWD-Stacking Ensemble Model for Blood Glucose Prediction

This module implements the complete AWD-stacking (Adaptive Weighted Deep Stacking)
ensemble learning algorithm as described in the paper.

Architecture (Fig. 6):
1. Base estimators: BiLSTM, StackLSTM, VanillaLSTM
2. 5-fold cross-validation for base model training
3. Improved AP clustering for adaptive weighting
4. Linear regression as meta-model
5. Multi-history window technique

The meta-model training includes:
- Stacking ensemble learning output
- Adaptive weighted base learner outputs
- Original training set features
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import os

from models import BiLSTMModel, StackLSTMModel, VanillaLSTMModel, BaseLSTMModel
from clustering import AdaptiveWeightCalculator


class AWDStackingEnsemble:
    """
    Adaptive Weighted Deep Stacking Ensemble for blood glucose prediction.

    This ensemble model combines:
    1. Three improved LSTM base models (BiLSTM, StackLSTM, VanillaLSTM)
    2. Improved affinity propagation clustering for adaptive weighting
    3. Stacking ensemble with linear regression meta-model
    4. Multi-history window technique
    """

    def __init__(self,
                 history_windows: List[int] = [6, 9, 12, 18],
                 prediction_horizon: int = 6,
                 n_folds: int = 5,
                 alpha: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize AWD-Stacking Ensemble.

        Args:
            history_windows: List of history window sizes (in data points)
                            Default: [6, 9, 12, 18] for 30, 45, 60, 90 minutes
            prediction_horizon: Number of future timesteps to predict (6, 9, or 12)
            n_folds: Number of folds for cross-validation
            alpha: Weight coefficient for AP clustering
            random_seed: Random seed for reproducibility
        """
        self.history_windows = history_windows
        self.prediction_horizon = prediction_horizon
        self.n_folds = n_folds
        self.alpha = alpha
        self.random_seed = random_seed

        # Base models for each history window
        self.base_models: Dict[int, List[BaseLSTMModel]] = {}

        # Adaptive weight calculator
        self.weight_calculator = AdaptiveWeightCalculator(alpha=alpha)

        # Meta-model (Linear Regression)
        self.meta_model = LinearRegression()

        # Store adaptive weights for each history window
        self.adaptive_weights: Dict[int, np.ndarray] = {}

        # Training state
        self.is_fitted = False

    def _create_base_models(self, history_window: int) -> List[BaseLSTMModel]:
        """
        Create the three base models for a given history window.

        Args:
            history_window: Number of input timesteps

        Returns:
            List of [BiLSTM, StackLSTM, VanillaLSTM] models
        """
        return [
            BiLSTMModel(
                history_window=history_window,
                prediction_horizon=self.prediction_horizon,
                units=128,
                random_seed=self.random_seed
            ),
            StackLSTMModel(
                history_window=history_window,
                prediction_horizon=self.prediction_horizon,
                units=[128, 64, 32],
                dropout_rate=0.2,
                random_seed=self.random_seed + 1
            ),
            VanillaLSTMModel(
                history_window=history_window,
                prediction_horizon=self.prediction_horizon,
                units=128,
                use_peephole=True,
                random_seed=self.random_seed + 2
            )
        ]

    def _train_base_models_cv(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              history_window: int,
                              epochs: int = 500,
                              batch_size: int = 32,
                              patience: int = 10,
                              verbose: int = 0) -> Tuple[List[BaseLSTMModel], np.ndarray, np.ndarray]:
        """
        Train base models using 5-fold cross-validation.

        The stacking approach trains base learners on the original data,
        and their predictions serve as new features for the meta-model.

        Args:
            X: Input sequences (n_samples, history_window, 1)
            y: Target sequences (n_samples, prediction_horizon)
            history_window: History window size
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level

        Returns:
            trained_models: List of trained base models
            train_meta_features: Predictions on training data (for meta-model)
            test_meta_features: Averaged predictions on test data
        """
        n_samples = len(X)
        n_models = 3  # BiLSTM, StackLSTM, VanillaLSTM

        # Storage for meta-features
        train_meta_features = np.zeros((n_samples, self.prediction_horizon * n_models))
        fold_test_preds = [[] for _ in range(n_models)]

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)

        # Train models with cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            if verbose:
                print(f"    Fold {fold + 1}/{self.n_folds}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Create fresh models for each fold
            fold_models = self._create_base_models(history_window)

            for model_idx, model in enumerate(fold_models):
                model.build()
                model.train(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    verbose=0
                )

                # Get predictions for validation fold (training set of meta-model)
                val_preds = model.predict(X_val_fold)
                start_col = model_idx * self.prediction_horizon
                end_col = start_col + self.prediction_horizon
                train_meta_features[val_idx, start_col:end_col] = val_preds

        # Train final models on full training data
        final_models = self._create_base_models(history_window)

        # Split for final training (80/20)
        split_idx = int(len(X) * 0.8)
        X_train_final, X_val_final = X[:split_idx], X[split_idx:]
        y_train_final, y_val_final = y[:split_idx], y[split_idx:]

        for model in final_models:
            model.build()
            model.train(
                X_train_final, y_train_final,
                X_val_final, y_val_final,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=0
            )

        return final_models, train_meta_features, y

    def fit(self,
            X_dict: Dict[int, np.ndarray],
            y: np.ndarray,
            X_original: Optional[np.ndarray] = None,
            epochs: int = 500,
            batch_size: int = 32,
            patience: int = 10,
            verbose: int = 1) -> 'AWDStackingEnsemble':
        """
        Fit the AWD-Stacking ensemble model.

        Args:
            X_dict: Dictionary mapping history_window -> input sequences
                   Each array shape: (n_samples, history_window, 1)
            y: Target sequences (n_samples, prediction_horizon)
            X_original: Original training features for meta-model (optional)
            epochs: Maximum training epochs for base models
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level

        Returns:
            self
        """
        if verbose:
            print("Training AWD-Stacking Ensemble...")

        all_meta_features = []
        all_base_predictions = []

        # Train base models for each history window
        for hw in self.history_windows:
            if hw not in X_dict:
                raise ValueError(f"Missing data for history window {hw}")

            X_hw = X_dict[hw]

            if verbose:
                print(f"\n  Training models for history window {hw}...")

            # Train base models with cross-validation
            models, meta_features, _ = self._train_base_models_cv(
                X_hw, y,
                history_window=hw,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose
            )

            self.base_models[hw] = models
            all_meta_features.append(meta_features)

            # Get base predictions for adaptive weighting
            base_preds = [m.predict(X_hw) for m in models]
            all_base_predictions.extend(base_preds)

        # Compute adaptive weights using improved AP clustering
        if verbose:
            print("\n  Computing adaptive weights...")

        # Stack all base predictions for weighting
        stacked_preds = np.array(all_base_predictions)
        self.global_weights = self.weight_calculator.compute_weights(
            all_base_predictions, y
        )

        # Store weights per history window
        n_models_per_hw = 3
        for i, hw in enumerate(self.history_windows):
            start_idx = i * n_models_per_hw
            end_idx = start_idx + n_models_per_hw
            hw_weights = self.global_weights[start_idx:end_idx]
            # Renormalize within window
            self.adaptive_weights[hw] = hw_weights / np.sum(hw_weights)

        # Prepare meta-model training data
        if verbose:
            print("\n  Training meta-model...")

        # Combine all meta-features
        meta_X = np.hstack(all_meta_features)

        # Add weighted predictions
        weighted_preds = np.zeros((len(y), self.prediction_horizon))
        weight_idx = 0
        for hw in self.history_windows:
            hw_models = self.base_models[hw]
            hw_weights = self.adaptive_weights[hw]
            X_hw = X_dict[hw]

            for model, w in zip(hw_models, hw_weights):
                weighted_preds += w * model.predict(X_hw)
                weight_idx += 1

        # Normalize by total weight
        weighted_preds /= len(self.history_windows)
        meta_X = np.hstack([meta_X, weighted_preds])

        # Add original training features if provided
        if X_original is not None:
            meta_X = np.hstack([meta_X, X_original])

        # Train meta-model (Linear Regression)
        self.meta_model.fit(meta_X, y)

        self.is_fitted = True

        if verbose:
            print("\n  Training complete!")

        return self

    def predict(self,
                X_dict: Dict[int, np.ndarray],
                X_original: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained ensemble.

        Args:
            X_dict: Dictionary mapping history_window -> input sequences
            X_original: Original features for meta-model (optional)

        Returns:
            Predictions of shape (n_samples, prediction_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples = list(X_dict.values())[0].shape[0]

        # Get base model predictions
        all_meta_features = []
        weighted_preds = np.zeros((n_samples, self.prediction_horizon))

        for hw in self.history_windows:
            if hw not in X_dict:
                raise ValueError(f"Missing data for history window {hw}")

            X_hw = X_dict[hw]
            hw_models = self.base_models[hw]
            hw_weights = self.adaptive_weights[hw]

            hw_meta_features = []

            for model, w in zip(hw_models, hw_weights):
                preds = model.predict(X_hw)
                hw_meta_features.append(preds)
                weighted_preds += w * preds

            # Flatten for meta-model
            hw_meta = np.hstack(hw_meta_features)
            all_meta_features.append(hw_meta)

        # Normalize weighted predictions
        weighted_preds /= len(self.history_windows)

        # Prepare meta-model input
        meta_X = np.hstack(all_meta_features)
        meta_X = np.hstack([meta_X, weighted_preds])

        if X_original is not None:
            meta_X = np.hstack([meta_X, X_original])

        # Meta-model prediction
        predictions = self.meta_model.predict(meta_X)

        return predictions

    def save(self, filepath: str):
        """
        Save the ensemble model to disk.

        Args:
            filepath: Path to save the model
        """
        # Save base models
        os.makedirs(filepath, exist_ok=True)

        for hw, models in self.base_models.items():
            for i, model in enumerate(models):
                model_path = os.path.join(filepath, f"base_model_hw{hw}_m{i}.keras")
                model.model.save(model_path)

        # Save meta-model and weights
        meta_path = os.path.join(filepath, "meta_model.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'meta_model': self.meta_model,
                'adaptive_weights': self.adaptive_weights,
                'global_weights': self.global_weights,
                'history_windows': self.history_windows,
                'prediction_horizon': self.prediction_horizon,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'AWDStackingEnsemble':
        """
        Load a saved ensemble model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded AWDStackingEnsemble instance
        """
        import tensorflow as tf

        # Load meta-model and config
        meta_path = os.path.join(filepath, "meta_model.pkl")
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)

        ensemble = cls(
            history_windows=data['history_windows'],
            prediction_horizon=data['prediction_horizon']
        )
        ensemble.meta_model = data['meta_model']
        ensemble.adaptive_weights = data['adaptive_weights']
        ensemble.global_weights = data['global_weights']

        # Load base models
        for hw in data['history_windows']:
            models = []
            for i in range(3):  # 3 base models
                model_path = os.path.join(filepath, f"base_model_hw{hw}_m{i}.keras")
                keras_model = tf.keras.models.load_model(model_path)

                # Wrap in appropriate class
                if i == 0:
                    wrapper = BiLSTMModel(hw, ensemble.prediction_horizon)
                elif i == 1:
                    wrapper = StackLSTMModel(hw, ensemble.prediction_horizon)
                else:
                    wrapper = VanillaLSTMModel(hw, ensemble.prediction_horizon)

                wrapper.model = keras_model
                models.append(wrapper)

            ensemble.base_models[hw] = models

        ensemble.is_fitted = True
        return ensemble


class SimplifiedAWDStacking:
    """
    Simplified AWD-Stacking for single history window.

    This is a streamlined version for easier understanding and debugging.
    """

    def __init__(self,
                 history_window: int = 12,
                 prediction_horizon: int = 6,
                 n_folds: int = 5,
                 random_seed: int = 42):
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        self.n_folds = n_folds
        self.random_seed = random_seed

        self.base_models: List[BaseLSTMModel] = []
        self.meta_model = LinearRegression()
        self.weights = None
        self.is_fitted = False

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 500,
            batch_size: int = 32,
            patience: int = 10,
            verbose: int = 1) -> 'SimplifiedAWDStacking':
        """
        Fit the simplified ensemble.

        Args:
            X: Input sequences (n_samples, history_window, 1)
            y: Target sequences (n_samples, prediction_horizon)
        """
        n_samples = len(X)
        n_models = 3

        # Storage for out-of-fold predictions
        oof_preds = np.zeros((n_samples, self.prediction_horizon * n_models))

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)

        if verbose:
            print("Training base models with cross-validation...")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            if verbose:
                print(f"  Fold {fold + 1}/{self.n_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and train models
            fold_models = [
                BiLSTMModel(self.history_window, self.prediction_horizon,
                           random_seed=self.random_seed),
                StackLSTMModel(self.history_window, self.prediction_horizon,
                              random_seed=self.random_seed + 1),
                VanillaLSTMModel(self.history_window, self.prediction_horizon,
                                random_seed=self.random_seed + 2)
            ]

            for i, model in enumerate(fold_models):
                model.build()
                model.train(X_train, y_train, X_val, y_val,
                           epochs=epochs, batch_size=batch_size,
                           patience=patience, verbose=0)

                # Store OOF predictions
                val_pred = model.predict(X_val)
                start = i * self.prediction_horizon
                end = start + self.prediction_horizon
                oof_preds[val_idx, start:end] = val_pred

        # Train final base models
        if verbose:
            print("\nTraining final base models...")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.base_models = [
            BiLSTMModel(self.history_window, self.prediction_horizon,
                       random_seed=self.random_seed),
            StackLSTMModel(self.history_window, self.prediction_horizon,
                          random_seed=self.random_seed + 1),
            VanillaLSTMModel(self.history_window, self.prediction_horizon,
                            random_seed=self.random_seed + 2)
        ]

        for model in self.base_models:
            model.build()
            model.train(X_train, y_train, X_val, y_val,
                       epochs=epochs, batch_size=batch_size,
                       patience=patience, verbose=0)

        # Compute adaptive weights
        if verbose:
            print("\nComputing adaptive weights...")

        base_preds = [m.predict(X) for m in self.base_models]
        weight_calc = AdaptiveWeightCalculator()
        self.weights = weight_calc.compute_weights(base_preds, y)

        # Train meta-model
        if verbose:
            print("\nTraining meta-model...")

        self.meta_model.fit(oof_preds, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Get base predictions
        base_preds = [m.predict(X) for m in self.base_models]

        # Stack for meta-model
        meta_X = np.hstack(base_preds)

        # Meta-model prediction
        return self.meta_model.predict(meta_X)
