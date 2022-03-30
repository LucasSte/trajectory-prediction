# Trajectory Prediction
This repository introduces an encoder-decoder sequence to sequence neural network to forecast trajectories.
The neural network can be configured to a variable length input and predict a reasonable number of future time steps.

We utilized data from RoboCup SSL games to train the model. The inputs are a sequence of position, velocity and orientation 
for each time step, like the following:
```
[x y v_x v_y psi]
```

The network predicts a sequence of `[v_x v_y]`, which we integrate to find the
robot's future trajectory. A representation of the neural network is depicted below:

![alt text](https://github.com/LucasSte/trajectory-prediction/raw/master/docs/Robot_overview_nn.png)

We also analysed adding information about the ball, in an attempt to improve prediction. We conceived a ball encoder that
processes a sequence of position and velocity for the ball and aggregates that into the prediction. A diagram containing the architecture to aggregate 
information about the ball is available below:

![alt text](https://github.com/LucasSte/trajectory-prediction/raw/master/docs/ball_encoder.png)

#### More information

To find out more about the model's architecture, training and testing procedures, please check out
my [graduation thesis](https://github.com/LucasSte/Research/blob/4c6dd15c91670505114df42b3bab0490a8bf1844/tese.pdf).

### Running the models

#### Prepare the dataset
1. Enter the `dataset` folder, by doing `cd dataset`.
2. Run `download_dataset.sh`. This file downloads the dataset from RoboCup official logs repositories.
3. Run `python3 process_dataset.py` to process the dataset and prepare it for training and testing.

#### Train the models

From the root folder, run `python3 train_models.py`. It will train three models. 
* Two models that consume only data about the robots.
* Two models that consume data about tha ball and the robots.
* A multilayer perceptron network.

Each model has been trained in two configurations:
1. A look back window of 30 time steps and a prediction of 15 time steps.
2. A look back window of 60 time steps and a prediction of 30 time steps.

If you would like to visualize plots of batch error and validation error during training,
uncomment the `plot` function in `train_models.py` and in `comparison_tests.py`.

#### Testing the models

Running `python3 compare_models.py` will run all the trained configurations in a testing set.
It will calculate the mean average error, average displacement error and final displacement error and print them.
It will also measure such metrics for a Kalman predictor, which serves as a reference for comparison.
