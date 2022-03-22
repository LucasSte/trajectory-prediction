import os
from dataset.load_dataset import LoadDataSet
from ai_model.predictor import RobotOnlyPredictor, BallRobotPredictor
import tensorflow as tf
from ai_model.losses import SequenceLoss
from ai_model.batch_logs import BatchLogs
import matplotlib.pyplot as plt
from comparison_tests import MLPComparison


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
files = ['dataset/proc_set_1', 'dataset/proc_set_2']


def plots(logs):
    plt.figure()
    plt.plot(logs.batch_logs)
    plt.title('Training batch loss')
    plt.figure()
    plt.plot(logs.val_logs)
    plt.title('Validation loss')
    plt.show()


def train_models(look_back, look_forth, output_dims, robot_model_name, ball_model_name):
    loader = LoadDataSet(look_back, look_forth)
    robot_x, ball_x, ball_mask, y = loader.load_data(files, for_test=False)

    lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=1000, decay_rate=0.98,
    )

    print(f'--- Training robot model {look_back} -> {look_forth}')
    seq_predictor = RobotOnlyPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule), loss=SequenceLoss(), run_eagerly=False)
    batch_logs = BatchLogs()
    seq_predictor.fit(robot_x, y, epochs=10, batch_size=2048, callbacks=[batch_logs], validation_split=0.1)
    seq_predictor.save_model(robot_model_name)
    #plots(batch_logs)

    print(f'--- Training ball model {look_back} -> {look_forth}')
    seq_predictor = BallRobotPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule), loss=SequenceLoss(), run_eagerly=False)
    batch_logs = BatchLogs()
    seq_predictor.fit([robot_x, ball_x, ball_mask], y, epochs=10, batch_size=2048, callbacks=[batch_logs], validation_split=0.1)
    seq_predictor.save_model(ball_model_name)
    #plots(batch_logs)


train_models(30, 15, 2, 'robot_30_15', 'ball_30_15')
train_models(60, 30, 2, 'robot_30_60', 'ball_30_60')

# print('--- Training MLP model 15 -> 30')
# mlp_comparison_model = MLPComparison(30, 15, 2)
# mlp_comparison_model.train_model(files, 'mlp_comp')
#
# print('--- Training MLP model 30 -> 60')
# mlp_comparison_model = MLPComparison(60, 0, 2)
# mlp_comparison_model.train_model(files, 'mlp_comp_2')
