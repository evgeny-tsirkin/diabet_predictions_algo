"""
Improved Affinity Propagation Clustering for Adaptive Weighting

This module implements the weighted similarity matrix affinity propagation (AP)
clustering algorithm used to adaptively weight the base estimators in the
AWD-stacking ensemble model.

The algorithm:
1. Constructs a weighted similarity matrix using distance, variance, and weight coefficient
2. Uses AP clustering to calculate cluster centers and members
3. Assigns weights to base models based on clustering results
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.cluster import AffinityPropagation


class WeightedAffinityPropagation:
    """
    Improved Affinity Propagation clustering with weighted similarity matrix.

    This algorithm integrates similarity matrix and weight information to better
    represent relationships between data points (base model predictions) and
    improve the ensemble weighting.
    """

    def __init__(self,
                 alpha: float = 0.5,
                 damping: float = 0.5,
                 max_iter: int = 200,
                 convergence_iter: int = 15,
                 random_state: Optional[int] = 42):
        """
        Initialize Weighted Affinity Propagation.

        Args:
            alpha: Weight coefficient for variance term (Eq. 15)
            damping: Damping factor for AP algorithm (0.5 to 1.0)
            max_iter: Maximum number of iterations
            convergence_iter: Number of iterations with no change to stop
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.random_state = random_state

    def compute_weighted_similarity_matrix(self,
                                           predictions: np.ndarray) -> np.ndarray:
        """
        Compute weighted similarity matrix between base model predictions.

        The similarity matrix is calculated based on Euclidean distance
        and variance with weight coefficient alpha (Eq. 15-17):

        S_{i,j} = -dist_{i,j} - α(var_i + var_j)

        where:
        - dist_{i,j} = Σ(x_{i,k} - x_{j,k})^2 (Eq. 16)
        - var_i = (1/n) * Σ(x_{i,k} - x̄_i)^2 (Eq. 17)

        Args:
            predictions: Array of shape (n_models, n_samples, n_features)
                        or (n_models, n_samples) containing predictions
                        from each base model

        Returns:
            Weighted similarity matrix of shape (n_models, n_models)
        """
        n_models = predictions.shape[0]

        # Flatten predictions if multi-dimensional
        if predictions.ndim > 2:
            predictions = predictions.reshape(n_models, -1)

        # Compute variance for each model's predictions (Eq. 17)
        variances = np.var(predictions, axis=1)

        # Compute pairwise distances (Eq. 16)
        similarity = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    # Self-similarity (preference)
                    similarity[i, j] = -np.median(variances)
                else:
                    # Euclidean distance squared
                    dist_ij = np.sum((predictions[i] - predictions[j]) ** 2)

                    # Weighted similarity (Eq. 15)
                    # S_{i,j} = -dist_{i,j} - α(var_i + var_j)
                    similarity[i, j] = -dist_ij - self.alpha * (variances[i] + variances[j])

        return similarity

    def compute_cluster_variance_weights(self,
                                         predictions: np.ndarray,
                                         labels: np.ndarray,
                                         cluster_centers_indices: np.ndarray) -> np.ndarray:
        """
        Compute weights based on weighted variance sum of clustering results.

        Implements Eq. 18-19:
        w_k = Σ_{i=1}^{N} (1/C) * Σ_{j∈c_i} (x_{j,k} - μ_{i,k})^2
        w_k = w_k / Σ_{j=1}^{M} w_j  (normalization)

        Args:
            predictions: Predictions from base models (n_models, n_samples)
            labels: Cluster labels for each model
            cluster_centers_indices: Indices of cluster centers

        Returns:
            Normalized weights for each base model
        """
        n_models = predictions.shape[0]

        if predictions.ndim > 2:
            predictions = predictions.reshape(n_models, -1)

        n_features = predictions.shape[1]
        n_clusters = len(np.unique(labels))

        # Compute weighted variance sum for each feature
        feature_weights = np.zeros(n_features)

        for k in range(n_features):
            weighted_var_sum = 0
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_predictions = predictions[cluster_mask, k]

                if len(cluster_predictions) > 0:
                    cluster_mean = np.mean(cluster_predictions)
                    # Weighted variance (Eq. 18)
                    weighted_var_sum += (1.0 / n_clusters) * np.sum(
                        (cluster_predictions - cluster_mean) ** 2
                    )

            feature_weights[k] = weighted_var_sum

        # Compute model weights based on cluster membership and center proximity
        model_weights = np.zeros(n_models)

        for i in range(n_models):
            if i in cluster_centers_indices:
                # Cluster centers get higher weight
                model_weights[i] = 1.0 / (1.0 + np.var(predictions[i]))
            else:
                # Non-centers get weight based on distance to their cluster center
                center_idx = cluster_centers_indices[labels[i]]
                dist_to_center = np.sum((predictions[i] - predictions[center_idx]) ** 2)
                model_weights[i] = 1.0 / (1.0 + dist_to_center + np.var(predictions[i]))

        # Normalize weights (Eq. 19)
        model_weights = model_weights / np.sum(model_weights)

        return model_weights

    def fit_predict(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit AP clustering and compute adaptive weights for base models.

        Args:
            predictions: Predictions from base models
                        Shape: (n_models, n_samples) or (n_models, n_samples, n_features)

        Returns:
            labels: Cluster labels for each model
            cluster_centers_indices: Indices of cluster centers
            weights: Normalized weights for each base model
        """
        # Compute weighted similarity matrix
        similarity = self.compute_weighted_similarity_matrix(predictions)

        # Run Affinity Propagation clustering
        ap = AffinityPropagation(
            affinity='precomputed',
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            random_state=self.random_state
        )

        labels = ap.fit_predict(similarity)
        cluster_centers_indices = ap.cluster_centers_indices_

        # Handle case where AP doesn't converge (all same cluster)
        if cluster_centers_indices is None or len(cluster_centers_indices) == 0:
            labels = np.zeros(predictions.shape[0], dtype=int)
            cluster_centers_indices = np.array([0])

        # Compute adaptive weights
        weights = self.compute_cluster_variance_weights(
            predictions, labels, cluster_centers_indices
        )

        return labels, cluster_centers_indices, weights


class AdaptiveWeightCalculator:
    """
    Calculate adaptive weights for base model predictions using
    improved affinity propagation clustering.

    This class wraps the WeightedAffinityPropagation to provide a
    simple interface for the AWD-stacking ensemble.
    """

    def __init__(self,
                 alpha: float = 0.5,
                 use_prediction_variance: bool = True,
                 use_error_based_weights: bool = True):
        """
        Initialize adaptive weight calculator.

        Args:
            alpha: Weight coefficient for variance in similarity calculation
            use_prediction_variance: Include prediction variance in weighting
            use_error_based_weights: Adjust weights based on validation errors
        """
        self.alpha = alpha
        self.use_prediction_variance = use_prediction_variance
        self.use_error_based_weights = use_error_based_weights
        self.wap = WeightedAffinityPropagation(alpha=alpha)

    def compute_weights(self,
                        base_predictions: List[np.ndarray],
                        y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute adaptive weights for base model predictions.

        Args:
            base_predictions: List of predictions from each base model
                             Each array shape: (n_samples, prediction_horizon)
            y_true: True values for validation (optional, for error-based weighting)

        Returns:
            weights: Normalized weights for each base model
        """
        # Stack predictions: (n_models, n_samples, prediction_horizon)
        predictions = np.array(base_predictions)

        # Get clustering-based weights
        _, _, cluster_weights = self.wap.fit_predict(predictions)

        if self.use_error_based_weights and y_true is not None:
            # Compute prediction errors
            n_models = len(base_predictions)
            errors = np.zeros(n_models)

            for i, pred in enumerate(base_predictions):
                errors[i] = np.mean((pred - y_true) ** 2)  # MSE

            # Inverse error weighting
            error_weights = 1.0 / (errors + 1e-8)
            error_weights = error_weights / np.sum(error_weights)

            # Combine clustering and error-based weights
            weights = 0.5 * cluster_weights + 0.5 * error_weights
            weights = weights / np.sum(weights)
        else:
            weights = cluster_weights

        return weights

    def apply_weights(self,
                      base_predictions: List[np.ndarray],
                      weights: np.ndarray) -> np.ndarray:
        """
        Apply weights to combine base model predictions.

        Args:
            base_predictions: List of predictions from each base model
            weights: Weights for each model

        Returns:
            Weighted combination of predictions
        """
        weighted_sum = np.zeros_like(base_predictions[0])

        for pred, w in zip(base_predictions, weights):
            weighted_sum += w * pred

        return weighted_sum


def compute_similarity_matrix_simple(predictions: np.ndarray,
                                     alpha: float = 0.5) -> np.ndarray:
    """
    Simplified similarity matrix computation for debugging.

    Args:
        predictions: (n_models, n_samples)
        alpha: Variance weight

    Returns:
        Similarity matrix
    """
    n_models = predictions.shape[0]

    # Flatten if needed
    if predictions.ndim > 2:
        predictions = predictions.reshape(n_models, -1)

    # Variances
    variances = np.var(predictions, axis=1)

    # Similarity matrix
    S = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                S[i, j] = -np.median(variances)
            else:
                dist = np.sqrt(np.sum((predictions[i] - predictions[j]) ** 2))
                S[i, j] = -dist - alpha * (variances[i] + variances[j])

    return S
