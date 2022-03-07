import os
from dataset.load_dataset import LoadDataSet
from ai_model.predictor import RobotOnlyPredictor, BallRobotPredictor
from ai_model.losses import TestLoss
from comparison_tests import MLPComparison, KalmanFilterComparison


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
files = ['dataset/proc_set_3']


def compare_models(look_back, look_forth, output_dims, robot_model_name, ball_model_name):
    loader = LoadDataSet(look_back, look_forth)
    robot_x, ball_x, ball_mask, y = loader.load_data(files, for_test=True)
    loader.convert_to_real(y)

    seq_predictor = RobotOnlyPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor.load_model(robot_model_name)
    res = seq_predictor.predict(robot_x, batch_size=1024)
    y_pred_conv = loader.convert_batch(robot_x, res)

    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    print(f'--- Results for robot model {look_back} -> {look_forth}')
    test_loss.print_error()

    seq_predictor = BallRobotPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor.load_model(ball_model_name)
    res = seq_predictor.predict([robot_x, ball_x, ball_mask], batch_size=1024)
    y_pred_conv = loader.convert_batch(robot_x, res)

    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    print(f'--- Results for ball model {look_back} -> {look_forth}')
    test_loss.print_error()


compare_models(15, 30, 2, 'robot_30_15', 'ball_30_15')
compare_models(30, 60, 2, 'robot_30_60', 'ball_30_60')

print('--- Results for MLP model 15 -> 30')
mlp_comparison_model = MLPComparison(15, 30, 2)
mlp_comparison_model.test_model(files, 'mlp_comp')

print('--- Results for MLP model 30 -> 60')
mlp_comparison_model = MLPComparison(30, 60, 2)
mlp_comparison_model.test_model(files, 'mlp_comp_2')

kf_comp = KalmanFilterComparison(15, 30, 'dataset/data_set_3', 'dataset/position_series_params')
kf_comp.perform_test()

kf_comp = KalmanFilterComparison(30, 60, 'dataset/data_set_3', 'dataset/position_series_params')
kf_comp.perform_test()


