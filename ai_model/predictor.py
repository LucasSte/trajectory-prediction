from aggregator import BallAggregator, BallAggregatorInputs
import tensorflow as tf
from encoder import Encoder, BallEncoder, BallEncoderInput
from decoder import DecoderInitializer, Decoder, DecoderInput, DecoderState
from shape_checker import ShapeChecker


class Predictor(tf.keras.Model):

    def __init__(self, units, look_back, look_forth, result_dims, use_tf_function=True, forcing=True):
        super(Predictor, self).__init__()
        self.look_forth = look_forth
        self.look_back = look_back
        self.result_dims = result_dims
        self.encoder = Encoder(units)
        self.decoder_init = DecoderInitializer(units)
        self.decoder = Decoder(units, result_dims)
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker()
        self.forcing = forcing

    def _loop_step_no_forcing(self, input_pos, target_pos, enc_output, dec_state):
        decoder_input = DecoderInput(
            new_tokens=input_pos,
            enc_output=enc_output
        )

        decoder_output = self.decoder(decoder_input, dec_state)
        self.shape_checker(decoder_output.sequence, ('batch', 't'))
        target_pos = tf.keras.layers.Reshape((self.result_dims,))(target_pos)
        self.shape_checker(target_pos, ('batch', 't'))
        dec_state = DecoderState(state_h=decoder_output.state_h, state_c=decoder_output.state_c)

        return dec_state, tf.keras.layers.Reshape((1, self.result_dims))(decoder_output.sequence)

    def _loop_step(self, new_seq, enc_output, dec_state):
        input_pos, target_pos = new_seq[:, 0:1, :], new_seq[:, 1:2, :]
        dec_state, seq = self._loop_step_no_forcing(input_pos, target_pos, enc_output, dec_state)

        return dec_state, seq

    def _train_step(self, data):
        pass

    @tf.function
    def _tf_train_step(self, data):
        return self._train_step(data)

    def train_step(self, data):
        self.shape_checker = ShapeChecker()
        if self.use_tf_function:
            return self._tf_train_step(data)
        else:
            return self._train_step(data)

    def save_model(self, name):
        self.save_weights(name, save_format='tf')

    def load_model(self, name):
        self.load_weights(name).expect_partial()

    def _forecast(self, input_seq):
        pass

    @tf.function
    def _tf_forecast(self, input_seq):
        return self._forecast(input_seq)

    def forecast(self, input_seq):
        if self.use_tf_function:
            return self._tf_forecast(input_seq)
        else:
            return self._forecast(input_seq)

    def call(self, inputs, training=None, mask=None):
        if self.use_tf_function:
            return self._tf_forecast(inputs)

        return self._forecast(inputs)


class RobotOnlyPredictor(Predictor):
    def __init__(self, units, look_back, look_forth, result_dims, use_tf_function=True, forcing=True):
        super(RobotOnlyPredictor, self).__init__(units, look_back, look_forth, result_dims, use_tf_function, forcing)

    def _train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            enc_output = self.encoder(x)

            dec_state = self.decoder_init(DecoderState(state_h=enc_output.state_hb, state_c=enc_output.state_cb))
            loss = tf.constant(0.0)

            dec_state, seq = self._loop_step(
                tf.stack([x[:, self.look_back - 1, 2:(4 + self.use_psi)], y[:, 0:]], axis=1),
                enc_output.state_h, dec_state)
            if self.forcing:
                for t in range(1, self.look_forth):
                    new_seq = y[:, (t - 1):(t + 1), :]
                    dec_state, point = self._loop_step(new_seq, enc_output.state_h, dec_state)
                    seq = tf.concat([seq, point], axis=1)
                loss = self.loss(y, seq)
            else:
                input_seq = seq
                for t in range(1, self.look_forth):
                    target_seq = y[:, (t - 1):t, :]
                    dec_state, point = self._loop_step_no_forcing(input_seq, target_seq, enc_output.state_h, dec_state)
                    input_seq = point
                    seq = tf.concat([seq, point], axis=1)
                loss = self.loss(y, seq)

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'batch_loss': loss}

    def _forecast(self, input_seq):
        enc_output = self.encoder(input_seq)
        new_tokens = input_seq[:, (self.look_back - 1):self.look_back, 2:4]
        dec_state = self.decoder_init(DecoderState(state_h=enc_output.state_hb, state_c=enc_output.state_cb))

        result_tokens = []
        for _ in range(self.look_forth):
            dec_input = DecoderInput(
                new_tokens=new_tokens, enc_output=enc_output.state_h
            )
            dec_output = self.decoder(dec_input, dec_state)
            dec_state = DecoderState(
                state_h=dec_output.state_h,
                state_c=dec_output.state_c,
            )
            new_tokens = tf.expand_dims(dec_output.sequence, axis=1)
            result_tokens.append(dec_output.sequence)

        result_tokens = tf.stack(result_tokens, axis=1)
        return result_tokens


class BallRobotPredictor(Predictor):
    def __init__(self, units, look_back, look_forth, result_dims, use_tf_function=True, forcing=True):
        super(BallRobotPredictor, self).__init__(units, look_back, look_forth, result_dims, use_tf_function, forcing)
        self.ball_encoder = BallEncoder(units)
        self.ball_aggregator = BallAggregator(units)

    def _loop_step_ball(self, new_seq, enc_output, dec_state, ball_pos):
        input_pos, target_pos = new_seq[:, 0:1, :], new_seq[:, 1:2, :]
        return self._loop_step_no_forcing_ball(input_pos, target_pos, enc_output, dec_state, ball_pos)

    def _loop_step_no_forcing_ball(self, input_pos, target_pos, enc_output, dec_state, ball_pos):
        decoder_input = DecoderInput(
            new_tokens=tf.concat([input_pos, ball_pos], axis=2),
            enc_output=enc_output
        )
        decoder_output = self.decoder(decoder_input, dec_state)
        # TODO: Simplify decoder step

    def _train_step(self, data):
        x, y = data
        [robot_data, ball_seq, ball_mask] = x

        with tf.GradientTape() as tape:
            robot_enc_output = self.encoder(robot_data)
            ball_enc_output = self.ball_encoder(BallEncoderInput(
                sequence=ball_seq, mask=ball_mask
            ))

            enc_context = self.ball_aggregator(BallAggregatorInputs(
                robot_seq=robot_enc_output.state_h, ball_seq=ball_enc_output.state_h
            ))

            dec_state = self.decoder_init(DecoderState(state_h=robot_enc_output.state_hb,
                                                       state_c=robot_enc_output.state_cb))
            loss = tf.constant(0.0)

            dec_state, seq = self._loop_step(
                tf.stack([robot_data[:, self.look_back-1, 2:4], y[:, 0, :]], axis=1), enc_context, dec_state,
                ball_enc_output.pos
            )

