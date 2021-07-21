from __future__ import annotations

import math
from datetime import datetime, time, timedelta
from typing import Any

import tensorflow.keras as keras
import matplotlib.pyplot as plt

# from .plotter import Plotter


class CustomCallback(keras.callbacks.Callback):
    def __init__(
            self: CustomCallback,
            num_epochs: int, epoch_interval: int = None,
            plotter: Plotter = None,
            batch_size: int = None,
            training_data: Any = None, validation_data: Any = None) -> None:
        self.num_epochs = num_epochs
        self.epoch_interval = max(1, math.floor(
            num_epochs / 10)) if epoch_interval is None else epoch_interval

        # Learning curve stuff etc.
        self.last_loss: float = None
        self.last_timestamp: datetime = None

        # Plotting learning journey
        self.plotter = plotter
        self.training_data = training_data
        self.validation_data = validation_data

        #
        if self.training_data is not None or self.validation_data is not None:
            if batch_size is None:
                raise 'batch_size may not remain unspecified if training or validation data is given'
        self.batch_size = batch_size

    def on_train_begin(self: CustomCallback, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("Starting training; got log keys: {}".format(keys))
        self.last_timestamp = datetime.now()
        pass

    def on_train_end(self: CustomCallback, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("Stop training; got log keys: {}".format(keys))
        pass

    def on_epoch_begin(self: CustomCallback, epoch: int, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # if epoch % self.epoch_interval == 0:
        #     print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        pass

    def on_epoch_end(self: CustomCallback, epoch: int, logs: dict = None) -> None:
        if epoch % self.epoch_interval == 0:
            percentage, loss_improvement = self._print_info_strings(epoch, logs)
            self._get_and_plot_train_preds(percentage, logs['loss'], loss_improvement)
            self._get_and_plot_val_preds(percentage, logs['val_loss'], loss_improvement)

    def on_test_begin(self: CustomCallback, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("Start testing; got log keys: {}".format(logs['keys']))
        pass

    def on_test_end(self: CustomCallback, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("Stop testing; got log keys: {}".format(keys))
        pass

    def on_predict_begin(self: CustomCallback, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("Start predicting; got log keys: {}".format(keys))
        pass

    def on_predict_end(self: CustomCallback, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("Stop predicting; got log keys: {}".format(keys))
        pass

    def on_train_batch_begin(self: CustomCallback, batch: int, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
        pass

    def on_train_batch_end(self: CustomCallback, batch: int, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        pass

    def on_test_batch_begin(self: CustomCallback, batch: int, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
        pass

    def on_test_batch_end(self: CustomCallback, batch: int, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
        pass

    def on_predict_batch_begin(self: CustomCallback, batch: int, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
        pass

    def on_predict_batch_end(self: CustomCallback, batch: int, logs: dict = None) -> None:
        # keys = list(logs.keys())
        # print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
        pass

    def _update_time(self: CustomCallback) -> str:
        now: datetime = datetime.now()
        return_val: str = ''
        if self.last_timestamp is not None:
            time_diff: timedelta = now - self.last_timestamp

            seconds_str: str = str(time_diff.seconds)
            micros_str: str = str(time_diff.microseconds)[:2]
            return_val: str = f'{seconds_str}.{micros_str}'
        self.last_timestamp = now
        return return_val

    def _update_loss_improvement(self: CustomCallback, new_loss: float) -> str:
        return_val: str = ''
        if self.last_loss is not None:
            diff = self.last_loss - new_loss
            improvement: float = -round((diff / self.last_loss * 100), 2)
            is_pos = improvement >= 0
            return_val = f'{+improvement}' if is_pos else f'{improvement}'

        self.last_loss = new_loss
        return return_val

    def _print_info_strings(self: CustomCallback, epoch: int, logs: dict) -> Any:
        percentage = epoch / self.num_epochs * 100
        new_loss = logs["loss"]

        time_diff: str = self._update_time()
        loss_improvement = self._update_loss_improvement(new_loss)
        self.print_results(epoch, new_loss, loss_improvement, percentage, time_diff)

        return percentage, loss_improvement

    def _get_and_plot_train_preds(self: CustomCallback, percentage: float, loss: float, improvement: str) -> None:
        if self.training_data is not None and self.plotter is not None:
            preds = self.model.predict(self.training_data, batch_size=self.batch_size)

            plt.plot(self.training_data.reshape(-1), label='original')
            plt.plot(preds.reshape(-1), label='reconstruction')
            plt.legend()
            plt.title(f'Train reconstruction: loss {loss} | {improvement}')

            self.plotter(str(int(percentage)), '/training_progress/train_data', verbose=False)

    def _get_and_plot_val_preds(self: CustomCallback, percentage: float, loss: float, improvement: float) -> None:
        if self.validation_data is not None and self.plotter is not None:
            preds = self.model.predict(self.validation_data, batch_size=self.batch_size)

            plt.plot(self.validation_data.reshape(-1), label='original')
            plt.plot(preds.reshape(-1), label='reconstruction')
            plt.legend()
            plt.title(f'Val reconstruction: val loss {loss} | {improvement}')

            self.plotter(str(int(percentage)), '/training_progress/val_data', verbose=False)

    def print_results(self: CustomCallback, epoch: int, loss: float, improvement: str, percentage: float, time: float) -> None:
        epoch_string = f'End of epoch {epoch}'
        loss_string = f'loss {loss}'
        improvement_string = f'({improvement:>6}%)' if len(improvement) > 0 else ''
        percentage_string = f'{percentage} % done'
        time_string = f'took {time} seconds'

        print(f'{epoch_string:<20} {loss_string:<30} {improvement_string:<40} {percentage_string:>13} {time_string:>35}')

# if __name__ == '__main__':
#     epochs = [0, 1, 2, 3, 4]
#     losses = [0.0056587159633636475, 0.0009753100457601249, 0.000884634384419769, 0.000861109234392643, 0.0008480784017592669]
#     improvements = ['', '-82.76', '-9.3', '-2.66', '-1.51']
#     percentages = [0.0, 10.0, 20.0, 30.0, 40.0]
#     times = [27.52, 32.48, 29.64, 29.71, 26.44]
#     for epoch, loss, improvement, percentage, time_diff in zip(epochs, losses, improvements, percentages, times):
#         print_results(epoch, loss, improvement, percentage, time_diff)
