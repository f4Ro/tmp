import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from numpy.random import seed  # isort:skip

seed(1)

from tensorflow.random import set_seed  # isort:skip

set_seed(2)


# Data
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers

# Machine learning
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model, Sequential


class RNN:
    def __init__(self, num_layers=1, regularization=0.0, activation="relu", optimizer="adam"):
        self.num_layers = num_layers
        self.regularization = regularization
        self.activation = activation
        self.optimizer = optimizer

    def get_layer_sizes(self, dims, num_layers):
        dims_sq = dims ** 2
        step_size = int(dims_sq / (num_layers))

        layers = []
        for x in range(dims_sq, 0, -step_size):
            if x >= dims:
                layers.append(x)
        if len(layers) > 1 and layers[-1] != dims:
            layers[-1] = dims
        return layers

    def reshape_inputs(self, data, sample_size):
        """
        Reshape inputs for LSTMs to [n_data_samples, sample_size, n_features]

        :description:
        Keras LSTM layers expect (at least) three dimensional input.
        n_data_samples -- the number of samples that is trained on
        sample_size    -- the length of each sample (the time series is broken down so BP and gradients work properly)
                        In an autoencoder this co-determines the compression ratio
        n_features     -- number of dimensions of the data. Dictates the size of the inputs later on

        :parameters:
        data(np.ndarray) -- matrix that holds the reshapable data
        sample_size(int) -- length of each sample. If x is not divisible without remainder the remainder is thrown away.

        :returns:
        reshaped_data(np.ndarray) -- data of the shape dictated by sample_size
        """

        #
        remainder = data.shape[0] % sample_size
        limit = data.shape[0] - remainder
        data = data[:limit, :]
        n_samples = int(data.shape[0] / sample_size)

        n_dims = data.shape[1]
        reshaped_data = data.reshape(n_samples, sample_size, n_dims)

        return reshaped_data

    def create_model(
        self,
        train_data,
        sequence_length=20,
        num_layers=1,
        regularization=0.0,
        activation="relu",
        optimizer="adam",
    ):
        """
        arguments:
        num_layers # (1,3)
        regularization per layer # (0.0 - 0.5)
        activation per layer # [lrelu, relu, sigmoid, tanh]
        optimizer # [adam, rmsprop, sgd]

        returns:
        model
        """
        # Helper function
        def create_layer(dim, return_sequences=True):
            return LSTM(
                dim,
                activation=activation,
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.L2(regularization),
            )

        # Assertions
        assert activation in [
            "relu",
            "sigmoid",
            "tanh",
            "linear",
        ], f'Unsupported activation "{activation}"'
        assert optimizer in ["adam", "sgd", "rmsprop"], f'Unsupported optimizer "{optimizer}"'
        assert num_layers >= 1 and num_layers <= 6, "Has to have at least one layer and max 6"

        train_data = self.reshape_inputs(train_data, sequence_length)
        layer_dims = self.get_layer_sizes(train_data.shape[2], num_layers)

        inputs = Input(shape=(train_data.shape[1], train_data.shape[2]))
        x = inputs
        for dim in layer_dims[:-1]:
            x = create_layer(dim)(x)
        x = create_layer(layer_dims[-1], return_sequences=False)(x)

        #
        x = RepeatVector(train_data.shape[1])(x)

        #
        for dim in reversed(layer_dims):
            x = create_layer(dim)(x)

        outputs = TimeDistributed(Dense(train_data.shape[2]))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss="mse")

        return model, train_data

    def train_model(self, model, data, num_epochs=50, batch_size=None, verbose=0):
        """
        Fit the model to encoding the training data.

        :parameters:
        data(np.ndarray) -- the training data
        num_epochs(int)  -- the number of times the model should iterate over the training data to learn
        batch_size(int)  -- the number of predictions the model makes before adjusting its weights

        :returns:
        history(keras.history) -- the history of the training, including loss and validation loss
        """

        history = model.fit(
            x=data,
            y=data,  # learns an identity function x_train is label and input
            epochs=num_epochs,
            shuffle=False,
            verbose=verbose,
        ).history

        return history

    def per_rms_diff(self, label, prediction):
        # assert prediction.dtype == tf.float32, f'wrong dtype for prediction {prediction.dtype}'
        # assert label.dtype == tf.float32, f'wrong dtype for label {prediction.dtype}'
        try:
            diff = np.sum(np.square(np.subtract(label, prediction)))
            sq = np.sum(np.square(label))
            prd = 100 * (np.sqrt(np.divide(diff, sq)))
        except ZeroDivisionError:
            print("Oh, no! You tried to divide by zero!")

        return prd

    def evaluate_model(self, x_test):
        x_test = self.reshape_inputs(x_test, 20)

        # Make predictions
        preds_test = self.model.predict(x_test)

        return self.per_rms_diff(x_test, preds_test)

    def fit(self, X, y=None):
        model, train_data = self.create_model(
            train_data=X,
            num_layers=self.num_layers,
            regularization=self.regularization,
            activation=self.activation,
            optimizer=self.optimizer,
        )

        self.model = model

        self.train_model(model, train_data)

        return self

    def score(self, X, y=None):
        score = self.evaluate_model(X)
        return score
