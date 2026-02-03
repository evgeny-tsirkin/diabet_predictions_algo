"""
Neural Network Models for AWD-stacking Blood Glucose Prediction

This module implements the three base estimators:
1. BiLSTM (Bidirectional Long Short-Term Memory)
2. StackLSTM (Stacked Long Short-Term Memory)
3. VanillaLSTM (LSTM with peephole connections)

All models follow the architecture described in the paper with:
- ReLU activation for LSTM layers
- Linear activation for dense output layer
- MSE loss function
- Adam optimizer with learning rate 0.001
"""

import numpy as np
from typing import Optional, Tuple, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping


class BaseLSTMModel:
    """Base class for LSTM models with common training interface."""

    def __init__(self,
                 history_window: int,
                 prediction_horizon: int,
                 learning_rate: float = 0.001,
                 random_seed: Optional[int] = 42):
        """
        Initialize base LSTM model.

        Args:
            history_window: Number of input timesteps
            prediction_horizon: Number of output timesteps
            learning_rate: Adam optimizer learning rate
            random_seed: Random seed for reproducibility
        """
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        if random_seed is not None:
            tf.random.set_seed(random_seed)
            np.random.seed(random_seed)

        self.model = None

    def build(self) -> Model:
        """Build and compile the model. To be implemented by subclasses."""
        raise NotImplementedError

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 500,
              batch_size: int = 32,
              patience: int = 10,
              verbose: int = 1) -> keras.callbacks.History:
        """
        Train the model with early stopping.

        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_val: Validation input sequences
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            self.build()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=verbose
        )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input sequences

        Returns:
            Predicted sequences
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model.predict(X, verbose=0)

    def get_model(self) -> Model:
        """Get the underlying Keras model."""
        if self.model is None:
            self.build()
        return self.model


class BiLSTMModel(BaseLSTMModel):
    """
    Bidirectional LSTM for blood glucose prediction.

    Architecture from the paper (Fig. 7):
    - Bidirectional LSTM layer with 128 units
    - Dense layer with N nodes (N = prediction_horizon)
    - Output layer with 1 node per timestep

    The BiLSTM processes inputs through forward and backward LSTM layers
    at each time step, concatenating the hidden states for the final output.
    """

    def __init__(self,
                 history_window: int,
                 prediction_horizon: int,
                 units: int = 128,
                 learning_rate: float = 0.001,
                 random_seed: Optional[int] = 42):
        """
        Initialize BiLSTM model.

        Args:
            history_window: Number of input timesteps
            prediction_horizon: Number of output timesteps
            units: Number of LSTM units
            learning_rate: Adam optimizer learning rate
            random_seed: Random seed for reproducibility
        """
        super().__init__(history_window, prediction_horizon, learning_rate, random_seed)
        self.units = units

    def build(self) -> Model:
        """Build and compile the BiLSTM model."""
        inputs = layers.Input(shape=(self.history_window, 1))

        # Bidirectional LSTM layer
        # Forward process equations (Eq. 10-14 in paper):
        # f_t = σ(W_f * x_t + U_f * h_{t-1} + b_f)  - Forget gate
        # i_t = σ(W_i * x_t + U_i * h_{t-1} + b_i)  - Input gate
        # o_t = σ(W_o * x_t + U_o * h_{t-1} + b_o)  - Output gate
        # c_t = f_t ⊙ c_{t-1} + i_t ⊙ relu(W_c * x_t + U_c * h_{t-1} + b_c)
        # h_t = o_t ⊙ relu(c_t)
        x = layers.Bidirectional(
            layers.LSTM(
                self.units,
                activation='relu',
                return_sequences=True
            )
        )(inputs)

        # Take the last timestep output
        x = layers.Lambda(lambda x: x[:, -1, :])(x)

        # Dense layer with N nodes
        x = layers.Dense(self.prediction_horizon, activation='relu')(x)

        # Output layer
        outputs = layers.Dense(self.prediction_horizon, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='BiLSTM')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return self.model


class StackLSTMModel(BaseLSTMModel):
    """
    Stacked LSTM for blood glucose prediction.

    Architecture from the paper (Fig. 8):
    - LSTM layer 1: 128 units with dropout
    - LSTM layer 2: 64 units with dropout
    - LSTM layer 3: 32 units
    - Dense layer with N nodes (prediction_horizon)

    StackLSTM stacks multiple LSTM networks in a hierarchical structure
    to create a deeper model capable of learning more complex patterns.
    """

    def __init__(self,
                 history_window: int,
                 prediction_horizon: int,
                 units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 random_seed: Optional[int] = 42):
        """
        Initialize StackLSTM model.

        Args:
            history_window: Number of input timesteps
            prediction_horizon: Number of output timesteps
            units: List of units for each LSTM layer
            dropout_rate: Dropout rate between layers
            learning_rate: Adam optimizer learning rate
            random_seed: Random seed for reproducibility
        """
        super().__init__(history_window, prediction_horizon, learning_rate, random_seed)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self) -> Model:
        """Build and compile the StackLSTM model."""
        inputs = layers.Input(shape=(self.history_window, 1))

        x = inputs

        # Stack LSTM layers
        for i, num_units in enumerate(self.units):
            return_sequences = (i < len(self.units) - 1)

            x = layers.LSTM(
                num_units,
                activation='linear',  # Paper uses linear activation for StackLSTM
                return_sequences=return_sequences
            )(x)

            # Add dropout between layers (except after last LSTM)
            if i < len(self.units) - 1:
                x = layers.Dropout(self.dropout_rate)(x)

        # Dense output layer
        outputs = layers.Dense(self.prediction_horizon, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='StackLSTM')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return self.model


class PeepholeLSTMCell(layers.Layer):
    """
    Custom LSTM cell with peephole connections.

    Peephole connections allow the gates to look at the cell state,
    which helps better handle long sequence data and capture
    long-distance dependencies.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]  # [h, c]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Input gate weights
        self.W_i = self.add_weight(shape=(input_dim, self.units), name='W_i')
        self.U_i = self.add_weight(shape=(self.units, self.units), name='U_i')
        self.P_i = self.add_weight(shape=(self.units,), name='P_i')  # Peephole
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')

        # Forget gate weights
        self.W_f = self.add_weight(shape=(input_dim, self.units), name='W_f')
        self.U_f = self.add_weight(shape=(self.units, self.units), name='U_f')
        self.P_f = self.add_weight(shape=(self.units,), name='P_f')  # Peephole
        self.b_f = self.add_weight(shape=(self.units,), initializer='ones', name='b_f')

        # Cell state weights
        self.W_c = self.add_weight(shape=(input_dim, self.units), name='W_c')
        self.U_c = self.add_weight(shape=(self.units, self.units), name='U_c')
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', name='b_c')

        # Output gate weights
        self.W_o = self.add_weight(shape=(input_dim, self.units), name='W_o')
        self.U_o = self.add_weight(shape=(self.units, self.units), name='U_o')
        self.P_o = self.add_weight(shape=(self.units,), name='P_o')  # Peephole
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')

        self.built = True

    def call(self, inputs, states):
        h_prev, c_prev = states

        # Forget gate with peephole connection
        f = tf.sigmoid(
            tf.matmul(inputs, self.W_f) +
            tf.matmul(h_prev, self.U_f) +
            c_prev * self.P_f +  # Peephole connection
            self.b_f
        )

        # Input gate with peephole connection
        i = tf.sigmoid(
            tf.matmul(inputs, self.W_i) +
            tf.matmul(h_prev, self.U_i) +
            c_prev * self.P_i +  # Peephole connection
            self.b_i
        )

        # Candidate cell state
        c_candidate = tf.tanh(
            tf.matmul(inputs, self.W_c) +
            tf.matmul(h_prev, self.U_c) +
            self.b_c
        )

        # New cell state
        c = f * c_prev + i * c_candidate

        # Output gate with peephole connection
        o = tf.sigmoid(
            tf.matmul(inputs, self.W_o) +
            tf.matmul(h_prev, self.U_o) +
            c * self.P_o +  # Peephole connection to new cell state
            self.b_o
        )

        # Hidden state
        h = o * tf.tanh(c)

        return h, [h, c]


class VanillaLSTMModel(BaseLSTMModel):
    """
    Vanilla LSTM with peephole connections for blood glucose prediction.

    Architecture from the paper (Fig. 9):
    - LSTM layer with 128 units and peephole connections
    - Dense layer with future data points as output nodes

    Peephole connections are added to the basic LSTM model to better
    handle long sequence data and capture long-distance dependencies.
    """

    def __init__(self,
                 history_window: int,
                 prediction_horizon: int,
                 units: int = 128,
                 learning_rate: float = 0.001,
                 use_peephole: bool = True,
                 random_seed: Optional[int] = 42):
        """
        Initialize VanillaLSTM model.

        Args:
            history_window: Number of input timesteps
            prediction_horizon: Number of output timesteps
            units: Number of LSTM units
            learning_rate: Adam optimizer learning rate
            use_peephole: Whether to use peephole connections
            random_seed: Random seed for reproducibility
        """
        super().__init__(history_window, prediction_horizon, learning_rate, random_seed)
        self.units = units
        self.use_peephole = use_peephole

    def build(self) -> Model:
        """Build and compile the VanillaLSTM model."""
        inputs = layers.Input(shape=(self.history_window, 1))

        if self.use_peephole:
            # Use custom peephole LSTM cell
            cell = PeepholeLSTMCell(self.units)
            x = layers.RNN(cell, return_sequences=False)(inputs)
        else:
            # Standard LSTM
            x = layers.LSTM(
                self.units,
                activation='linear',
                return_sequences=False
            )(inputs)

        # Dense output layer
        outputs = layers.Dense(self.prediction_horizon, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='VanillaLSTM')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return self.model


def create_base_models(history_window: int,
                       prediction_horizon: int,
                       random_seed: int = 42) -> List[BaseLSTMModel]:
    """
    Create all three base models for the ensemble.

    Args:
        history_window: Number of input timesteps
        prediction_horizon: Number of output timesteps
        random_seed: Random seed for reproducibility

    Returns:
        List of [BiLSTM, StackLSTM, VanillaLSTM] models
    """
    models = [
        BiLSTMModel(
            history_window=history_window,
            prediction_horizon=prediction_horizon,
            units=128,
            random_seed=random_seed
        ),
        StackLSTMModel(
            history_window=history_window,
            prediction_horizon=prediction_horizon,
            units=[128, 64, 32],
            dropout_rate=0.2,
            random_seed=random_seed + 1
        ),
        VanillaLSTMModel(
            history_window=history_window,
            prediction_horizon=prediction_horizon,
            units=128,
            use_peephole=True,
            random_seed=random_seed + 2
        )
    ]

    return models
