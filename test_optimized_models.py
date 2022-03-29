import time
from ai_model.predictor import RobotOnlyPredictor
from ai_model.losses import TestLoss
import os
import tensorflow as tf
import numpy as np
from dataset.load_dataset import LoadDataSet

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
models = ['robot_30_15_t', 'robot_30_60_t']
opt_models = ['opt_30_15', 'opt_60_30']
test_files = ['dataset/proc_set_3']


def optimize_model(list_idx, look_back, look_forth, input_dims, output_dims, batch_size):
    seq_predictor = RobotOnlyPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor.load_model(models[list_idx])
    full_model = tf.function(lambda x: seq_predictor(x))
    shape = (batch_size, look_back, input_dims)
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(shape, np.float32)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([full_model])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_quant_model = converter.convert()
    with open('./optimized_models/' + opt_models[list_idx] + str(batch_size) + '.tflite', 'wb') as f:
        f.write(tf_quant_model)


def common_step(list_idx, look_back, look_forth, batch_size):
    if not os.path.exists('./optimized_models/' + opt_models[list_idx] + str(batch_size) + '.tflite'):
        optimize_model(list_idx, look_back, look_forth, 5, 2, batch_size)
    interpreter = tf.lite.Interpreter('./optimized_models/' + opt_models[list_idx] + str(batch_size) + '.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    loader = LoadDataSet(look_back, look_forth)
    robot_x, _, _, y = loader.load_data(test_files, for_test=True)

    # ignore first inference
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return interpreter, input_details, output_details, robot_x, y, loader


def batch_inference_time(list_idx, look_back, look_forth, batch_size):
    interpreter, input_details, output_details, robot_x, _, _ = common_step(list_idx, look_back, look_forth, batch_size)

    start = time.time()
    for i in range(0, 100):
        input_data = np.float32(robot_x[(i*batch_size):((i+1)*batch_size)])
        # input_data = np.expand_dims(input_data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    print(f'Average inference time for model {look_back} -> {look_forth} of batch size {batch_size}')
    tt = (end-start)/100
    print(f'Time: {tt} s = {tt*1000} ms')


def score_test(list_idx, look_back, look_forth):
    interpreter, input_details, output_details, robot_x, y, loader = common_step(list_idx, look_back, look_forth, 1)

    res = []
    for i in range(0, np.shape(robot_x)[0]):
        input_data = np.float32(robot_x[i])
        input_data = np.expand_dims(input_data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        res.append(output_data[0])

    res = np.stack(res, axis=0)
    loader.convert_to_real(y)
    y_pred_conv = loader.convert_batch(robot_x, res)
    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    print(f'--- Results for robot model {look_back} -> {look_forth}')
    test_loss.print_error()


batch_inference_time(0, 30, 15, 1)
batch_inference_time(0, 30, 15, 11)
score_test(0, 30, 15)

batch_inference_time(1, 60, 30, 1)
batch_inference_time(1, 60, 30, 11)
score_test(1, 60, 30)

