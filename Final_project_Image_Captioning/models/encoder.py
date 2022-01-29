import tensorflow as tf


class CNN_Encoder(tf.keras.Model):
    def __init__(self, units):
        super(CNN_Encoder, self).__init__()
        self.fc_attn = tf.keras.layers.Dense(units)

    def call(self, f_for_attn):
        return tf.nn.relu(self.fc_attn(f_for_attn))  # [batch_size, 64, units]
