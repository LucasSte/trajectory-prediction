import tensorflow as tf
import numpy as np


class SequenceLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(SequenceLoss, self).__init__()
        self.name = 'seq_loss'
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.loss_2 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        return 100 * (self.loss(y_true, y_pred) + self.loss_2(y_true, y_pred))


class TestLoss:
    error = None

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        y_diff = np.abs(y_true - y_pred)
        y_sum = np.sum(y_diff, axis=2)
        y_sum_total = np.sum(y_sum, axis=1)
        return np.mean(y_sum_total)

    @staticmethod
    def average_displacement_error(y_true, y_pred):
        y_diff = y_true - y_pred
        y_sq = np.square(y_diff)
        y_dis = np.sqrt(np.sum(y_sq, axis=2))
        y_dis_total = np.sum(y_dis, axis=1)
        return np.mean(y_dis_total)

    @staticmethod
    def final_displacement_error(y_true, y_pred):
        y_diff = y_true[:, -1, :] - y_pred[:, -1, :]
        y_sq = np.square(y_diff)
        y_dis = np.sqrt(np.sum(y_sq, axis=1))
        return np.mean(y_dis)

    def __call__(self, y_true, y_pred):
        res = {
            'MAE': TestLoss.mean_absolute_error(y_true, y_pred),
            'ADE': TestLoss.average_displacement_error(y_true, y_pred),
            'FDE': TestLoss.final_displacement_error(y_true, y_pred)
        }

        self.error = res
        return res

    def print_error(self):
        if self.error is None:
            print("----Error not calculated----")
        else:
            for key, value in self.error.items():
                print(f'{key}: {value}')
