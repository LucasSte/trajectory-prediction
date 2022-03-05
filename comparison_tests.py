import os
import pickle
import numpy as np
from pykalman import KalmanFilter
from dataset.kalman_smoother import KalmanSmoother
from ai_model.losses import TestLoss, SequenceLoss
import tensorflow as tf
from dataset.load_dataset import LoadDataSet
import matplotlib.pyplot as plt


class KalmanFilterComparison:
    true = []
    predicted = []

    def __init__(self, look_back, look_forth, data_file, params_file):
        self.look_back = look_back
        self.look_forth = look_forth

        robots_f = open(data_file + '.pkl', 'rb')
        self.robots_t = pickle.load(robots_f)

        filter_params_f = open(params_file + '.pkl', 'rb')
        params = pickle.load(filter_params_f)
        self.transition_matrix = params.A
        self.observation_matrix = params.C

        self.observation_covariance = np.linalg.inv(np.matmul(params.V_neg_sqrt, params.V_neg_sqrt))
        self.transition_covariance = np.linalg.inv(np.matmul(params.W_neg_sqrt, params.W_neg_sqrt))

        self.smoother = KalmanSmoother()
        self.smoother.load_params(params_file)

    def get_future(self, a_matrix, last_pos):
        res = []
        for i in range(self.look_forth):
            pos = np.inner(a_matrix, last_pos)
            last_pos = pos
            res.append([pos[0], pos[2]])
        return res

    def process_robots(self, robots):
        for k in range(len(robots)):
            for robot_id, series in robots[k].items():
                if len(series['x']) > self.look_back:
                    x_sm, y_sm, _, _ = self.smoother.smooth(series['x'], series['y'], series['mask'])
                    ism = [series['x'][0], 0, series['y'], 0]
                    kf = KalmanFilter(transition_matrices=self.transition_matrix,
                                      observation_matrices=self.observation_matrix,
                                      initial_state_mean=ism,
                                      transition_covariance=self.transition_covariance)
                    initial = np.array((series['x'][0:self.look_back], series['y'][0:self.look_back])).T
                    means, cov = kf.filter(initial)

                    self.true.append(np.array((x_sm[self.look_back:(self.look_back + self.look_forth)],
                                               y_sm[self.look_back:(self.look_back + self.look_forth)])).T)
                    self.predicted.append(np.array(self.get_future(kf.transition_matrices, means[-1])))

                    means, cov = means[-1], cov[-1]

                    for i in range(self.look_back+1, len(series['x']-self.look_forth-1)):
                        self.true.append(np.array((x_sm[(i+1):(i+1+self.look_forth)],
                                                   y_sm[(i+1):(i+1+self.look_forth)])).T)
                        means, cov = kf.filter_update(means, cov,
                                                      np.array((series['x'][i], series['y'][i])))
                        self.predicted.append(np.array(self.get_future(kf.transition_matrices, means)))

    def perform_test(self):
        self.process_robots(self.robots_t['blue'])
        self.process_robots(self.robots_t['yellow'])

        true = np.array(self.true)
        predicted = np.array(self.predicted)
        loss = TestLoss()
        loss(true, predicted)
        print("----Kalman filter results----")
        print(f'Look back: {self.look_back} | Look forth: {self.look_forth}')
        loss.print_error()


class MLPBatchLogs(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MLPBatchLogs, self).__init__()
        self.batch_logs = []
        self.val_logs = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_logs.append(logs['loss'])

    def on_test_end(self, logs=None):
        self.val_logs.append(logs['loss'])


class MLPComparison:
    def __init__(self, look_back, look_forth, output_dims, use_cuda=True):
        self.look_back = look_back
        self.output_dims = output_dims
        self.look_forth = look_forth
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' if use_cuda else '-1'
        self.model = self.create_model()
        self.loader = LoadDataSet(look_back, look_forth)

    def create_model(self):
        data_input = tf.keras.Input(shape=(self.look_back, 5))
        x = tf.keras.layers.Dense(128, activation='relu')(data_input)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.output_dims*self.look_forth)(x)
        x = tf.keras.layers.Reshape((self.look_forth, self.output_dims))(x)

        return tf.keras.Model(inputs=data_input, outputs=x)

    def train_model(self, file_path: list):
        robot_x, _, _, y = self.loader.load_data(file_path)
        batch_logs = MLPBatchLogs()

        self.model.compile(optimizer=tf.optimizers.Adam(), loss=SequenceLoss(), run_eagerly=False)
        self.model.fit(robot_x, y, epochs=10, batch_size=1024, callbacks=[batch_logs], validation_split=0.1)

        plt.figure()
        plt.plot(batch_logs.batch_logs)
        plt.title('Batch loss during training')
        plt.plot(batch_logs.val_logs)
        plt.title('Batch loss during validation')

    def test_model(self, file_path: list):
        robot_x, _, _, y = self.loader.load_data(file_path, for_test=True)

        response = self.model.predict(robot_x)
        y_pred_conv = self.loader.convert_batch(robot_x, response)
        self.loader.convert_to_real(y)

        test_loss = TestLoss()
        test_loss(y[:, :, 0:2], y_pred_conv)
        test_loss.print_error()


