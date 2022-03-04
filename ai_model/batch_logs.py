import pickle
import tensorflow as tf


class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BatchLogs, self).__init__()
        self.batch_logs = []
        self.val_logs = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_logs.append(logs['batch_loss'])

    def on_test_end(self, logs=None):
        self.val_logs.append(logs['loss'])

    def save_vars(self, name):
        dic = {
            'batch': self.batch_logs,
            'val': self.val_logs,
        }

        with open('../saved_variables/' + name + '.pkl', 'wb') as f:
            pickle.dump(dic, f)