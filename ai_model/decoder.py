import typing
import tensorflow as tf
from .aggregator import AttentionAggregator2D
from .shape_checker import ShapeChecker


class DecoderInput(typing.NamedTuple):
    new_tokens: typing.Any
    enc_output: typing.Any


class DecoderState(typing.NamedTuple):
    state_h: typing.Any
    state_c: typing.Any


class DecoderOutput(typing.NamedTuple):
    sequence: typing.Any
    state_h: typing.Any
    state_c: typing.Any
    attention_weights: typing.Any


class DecoderInitializerInput(typing.NamedTuple):
    state_h: typing.Any
    state_c: typing.Any


class DecoderInitializer(tf.keras.layers.Layer):
    def __init__(self, dec_units):
        super(DecoderInitializer, self).__init__()
        self.fc_c = tf.keras.layers.Dense(dec_units, activation='tanh')
        self.fc_h = tf.keras.layers.Dense(dec_units, activation='tanh')

    def call(self, inputs: DecoderInitializerInput, state=None) -> DecoderState:
        return DecoderState(
            state_h=self.fc_h(inputs.state_h),
            state_c=self.fc_c(inputs.state_c)
        )


class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, seq_dim):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.seq_dim = seq_dim
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True)
        self.attention = AttentionAggregator2D(self.dec_units)
        self.fc = tf.keras.layers.Dense(seq_dim, activation='linear')

    def call(self, inputs: DecoderInput, state: DecoderState) -> DecoderOutput:
        shape_checker = ShapeChecker()
        shape_checker(inputs.new_tokens, ('batch', 't', 'dim'))
        shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))

        shape_checker(state.state_c, ('batch', 'dec_units'))
        shape_checker(state.state_h, ('batch', 'dec_units'))

        context_vector, attention_weights = self.attention(
            query=state.state_h, value=inputs.enc_output
        )

        context_vector = tf.keras.layers.Reshape((1, self.dec_units))(context_vector)
        lstm_input = tf.concat([inputs.new_tokens, context_vector], axis=2)

        lstm_output, state_h, state_c = self.lstm(lstm_input, initial_state=[state.state_h, state.state_c])
        sequence = self.fc(state_h)
        return DecoderOutput(
            sequence=sequence,
            state_h=state_h,
            state_c=state_c,
            attention_weights=attention_weights
        )
