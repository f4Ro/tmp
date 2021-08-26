# ==================================================================================================
# Imports etc.
# ==================================================================================================
#  External libraries
# Set seeds
from numpy.random import seed
from tensorflow.random import set_seed

seed(1)
set_seed(10)

import numpy as np
import matplotlib.pyplot as plt

# Tensorflow imports
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Input, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2, l1_l2

# Own code
from data_preprocessing.berkley_lab_data import read_and_preprocess_data
from shared_code.custom_callback import CustomCallback
import util.plotter as plotter
from util.terminal_colorizer import printc
from shared_code.per_rms_diff import per_rms_diff


# Plotting and visualization
# Machine learning


# Create instances of helpers
harry = plotter.Plotter("RNN", plt, backend="WebAgg")
# ==================================================================================================
# Data loading, preprocessing and plotting
# ==================================================================================================
batch_size: int = 1
sequence_length: int = 20
x_train, x_test = read_and_preprocess_data(
    sequence_length=sequence_length,
    batch_size=batch_size,
    motes_train=[7],
    motes_test=[7],
)
x_train = x_train[:1900, :, :]
x_test = x_test[1900:, :, :]
global_mean = np.mean(np.concatenate((x_train, x_test), axis=0))

printc(
    f"[train] NUM EXAMPLES | SEQUENCE LENGTH | NUM DIMENSIONS: {x_train.shape[0]} \
| {x_train.shape[1]} | {x_train.shape[2]}",
    color="blue",
)
printc(
    f"[test] NUM EXAMPLES | SEQUENCE LENGTH | NUM DIMENSIONS: {x_test.shape[0]} \
| {x_test.shape[1]} | {x_test.shape[2]}",
    color="blue",
)
printc("==========" * 13)
printc("Data preparation done -> model creation")
printc("==========" * 13)
# ==================================================================================================
# Building the model
# ==================================================================================================
# Configure the model
n_dims: int = x_train.shape[2]
num_epochs: int = 10
num_batches: int = int(x_train.shape[0] / batch_size)

global_mean_bias = Constant(global_mean)

# To be able to create a model elsewhere, have a function that can be imported
def create_model(
    sequence_length: int,
    n_dims: int,
    batch_size: int,
    activation: str = "tanh",
    recurrent_activation: str = "sigmoid",
    kernel_regularizer=l2(0.0),
    bias_regularizer=l2(0.0),
    activity_regularizer=l2(0.0),
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
):
    # [batch, timesteps, feature] is shape of inputs
    inputs: Input = Input(shape=(sequence_length, n_dims), batch_size=batch_size)
    x = inputs
    x = LSTM(
        n_dims,
        return_sequences=False,
        stateful=True,
        activation=activation,
        recurrent_activation=recurrent_activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
    )(x)
    x = RepeatVector(sequence_length)(x)
    x = LSTM(
        n_dims,
        stateful=True,
        return_sequences=True,
        bias_initializer=global_mean_bias,
        unit_forget_bias=False,
        activation=activation,
        recurrent_activation=recurrent_activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
    )(x)
    outputs = x

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


model = create_model(sequence_length, n_dims, batch_size)

printc(f"BATCH SIZE: {batch_size}", color="blue")
printc(f"COMPRESSION RATIO: {sequence_length}", color="blue")
printc(f"NUMBER OF WEIGHT UPDATES: {num_batches * num_epochs}", color="blue")
printc("==========" * 13)
printc("Model creation done -> training ")
printc("==========" * 13)
# ==================================================================================================
# Training the model
# ==================================================================================================
# Set up callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
own_callback = CustomCallback(
    num_epochs,
    plotter=harry,
    batch_size=batch_size,
    training_data=x_train,
    validation_data=x_test,
)
history = model.fit(
    x_train,
    x_train,
    epochs=num_epochs,
    batch_size=batch_size,
    shuffle=False,
    validation_data=(x_test, x_test),
    callbacks=[own_callback, early_stopping],
    verbose=0,
).history
printc("==========" * 13)
printc(f"Model training done ({num_epochs} epochs) -> Evaluation")
printc("==========" * 13)
# ==================================================================================================
# Evaluating the model
# ==================================================================================================
plt.plot(history["loss"], label="Train loss")
plt.plot(history["val_loss"], label="Validation loss")
plt.legend()
harry("loss", "/training_progress")

preds = model.predict(x_train, batch_size=batch_size)
plt.plot(x_train.reshape(-1), label="original")
plt.plot(preds.reshape(-1), label="reconstruction")
plt.legend()
harry("train_full", "/evaluation")
printc("PRMS-diff [train]: ", per_rms_diff(x_train, preds), color="yellow")

preds_test = model.predict(x_test, batch_size=batch_size)
plt.plot(x_test.reshape(-1), label="original")
plt.plot(preds_test.reshape(-1), label="reconstruction")
plt.legend()
harry("test_full", "/evaluation")
printc("PRMS-diff  [test]: ", per_rms_diff(x_test, preds_test), color="yellow")
printc("==========" * 13)
printc("Model evaluation done")
printc("==========" * 13)
# ==================================================================================================
# EOF
# ==================================================================================================
