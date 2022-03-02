import typing

import tensorflow as tf
from shape_checker import ShapeChecker


class AttentionAggregator(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionAggregator, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value):
        pass

    def continue_call(self, w1_query, value, shape_checker):
        shape_checker(value, ('batch', 's', 'value_units'))

        w2_key = self.W2(value)
        w2_key = self.W2(w2_key, ('batch', 's', 'attn_units'))

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            return_attention_scores=True,
        )

        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights


class AttentionAggregator2D(AttentionAggregator):
    def __init__(self, units):
        super(AttentionAggregator2D, self).__init__(units)

    def call(self, query, value):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 'query_units'))

        w1_query = self.W1(query)
        w1_query = tf.keras.layers.Reshape((1, self.units))(w1_query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))

        return self.continue_call(w1_query, value, shape_checker)


class AttentionAggregator3D(AttentionAggregator):
    def __init__(self, units):
        super(AttentionAggregator3D, self).__init__(units)

    def call(self, query, value):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))

        w1_query = self.W1(query)
        return self.continue_call(w1_query, value, shape_checker)


class BallAggregatorInputs(typing.NamedTuple):
    robot_seq: typing.Any
    ball_seq: typing.Any


class BallAggregator(tf.keras.layers.Layer):
    def __init__(self, units):
        self.attention = AttentionAggregator3D(units)
        self.dim = int(units*2/15)
        self.W1 = tf.keras.layers.Dense(self.dim)

    def call(self, inputs: BallAggregatorInputs, **kwargs):
        shape_checker = ShapeChecker()
        shape_checker(inputs.robot_seq, ('batch', 't', 'robot_dims'))

        context_vector, attention_weights = self.attention(
            query=inputs.ball_seq, value=inputs.robot_seq,
        )

        pos = self.W1(context_vector)
        pos = tf.keras.layers.Reshape((1, int(2*self.dim)))(pos)

        return pos
