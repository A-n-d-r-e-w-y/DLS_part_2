import tensorflow as tf


class CNNAttention(tf.keras.Model):
    def __init__(self, units):
        super(CNNAttention, self).__init__()
        self.fc_attn = tf.keras.layers.Dense(units)
        self.fc_state = tf.keras.layers.Dense(units)
        self.fc_v = tf.keras.layers.Dense(1)

    def call(self, f_for_attn, state):
        state = tf.expand_dims(state, 1)
        attn = (tf.nn.tanh(self.fc_attn(f_for_attn) + self.fc_state(state)))
        score = self.fc_v(attn)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * f_for_attn
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
