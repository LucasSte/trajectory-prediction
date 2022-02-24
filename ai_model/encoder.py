import typing
import tensorflow as tf
from shape_checker import ShapeChecker


class EncoderOutput(typing.NamedTuple):
    sequence: typing.Any
    state_h: typing.Any
    state_c: typing.Any
    state_hb: typing.Any
    state_cb: typing.Any


class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()

        self.lstm_config = tf.keras.layers.LSTM(enc_units, return_state=True, return_sequences=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(self.lstm_config)

    def call(self, sequence, state=None):
        shape_checker = ShapeChecker()
        shape_checker(sequence, ('batch', 'look_back', 'coordinates'))

        output, state_hf, state_cf, state_hb, state_cb = self.bi_lstm(sequence)
        shape_checker(state_hf, ('batch', 'enc_units'))
        shape_checker(state_cf, ('batch', 'enc_units'))
        shape_checker(state_hb, ('batch', 'enc_units'))
        shape_checker(state_cb, ('batch', 'enc_units'))

        state_h = tf.stack([state_hf, state_hb], axis=1)
        state_c = tf.stack([state_cf, state_cb], axis=1)

        return EncoderOutput(
            sequence=output,
            state_h=state_h,
            state_c=state_c,
            state_hb=state_hb,
            state_cb=state_cb,
        )


class BallEncoderInput(typing.NamedTuple):
    sequence: typing.Any
    mask: typing.Any


class BallEncoderOutput(typing.NamedTuple):
    sequence: typing.Any
    state_h: typing.Any
    state_c: typing.Any
    position_agg: typing.Any


class BallEncoder(tf.keras.layers.Layer):
    def __init__(self, enc_units):
        super(BallEncoder, self).__init__()
        self.enc_units = int(enc_units/2)
        self.lstm_config = tf.keras.layers.LSTM(self.enc_units, return_state=True, return_sequences=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(self.lstm_config)
        self.W1 = tf.keras.layers.Dense(4)

    def call(self, inputs: BallEncoderInput, state=None):
        shape_checker = ShapeChecker()
        shape_checker(inputs.sequence, ('batch', 'look_back', 'coordinates'))
        shape_checker(inputs.mask, ('batch', 'look_back'))

        output, state_hf, state_cf, state_hb, state_cb = self.bi_lstm(inputs.sequence, mask=inputs.mask)
        shape_checker(state_hf, ('batch', 'enc_units'))
        shape_checker(state_cf, ('batch', 'enc_units'))
        shape_checker(state_hb, ('batch', 'enc_units'))
        shape_checker(state_cb, ('batch', 'enc_units'))

        state_h = tf.stack([state_hf, state_hb], axis=1)
        state_c = tf.stack([state_cf, state_cb], axis=1)

        pos_agg = self.W1(state_h)
        pos_agg = tf.keras.layers.Reshape((1, 8))(pos_agg)
        return BallEncoderOutput(
            sequence=output,
            state_h=state_h,
            state_c=state_c,
            position_agg=pos_agg,
        )
