import tensorflow as tf

from models.attention import CNNAttention


class RNN_Attn_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, embedding_matrix=None):
        super(RNN_Attn_Decoder, self).__init__()
        self.units = units
        if embedding_matrix is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                       embeddings_initializer=tf.keras.initializers.Constant(
                                                           embedding_matrix),
                                                       trainable=True)
        self.rnn = tf.keras.layers.LSTMCell(units)
        self.fc = tf.keras.layers.Dense(units, activation='relu')
        self.fc_vocab = tf.keras.layers.Dense(vocab_size)
        self.fc_m0 = tf.keras.layers.Dense(units, activation='relu')
        self.fc_c0 = tf.keras.layers.Dense(units, activation='relu')
        self.attention = CNNAttention(units)

    def call(self, x, f_for_attn, state):
        # x: [batch_size] - targets
        # f_for_attn: [batch_size, 64, cnn_units]
        # state: [[batch_size, units], [batch_size, units]]
        context_vector, attention_weights = self.attention(f_for_attn, tf.concat((state),axis=1))  # [batch_size, units], # [batch_size, 64, 1]
        x = self.embedding(x)  # [batch_size, embedding_dim]
        x = tf.concat([context_vector, x], axis=-1)  # [batch_size, units + embedding_dim]
        output, state = self.rnn(x, state)  # [batch_size, units], [[batch_size, units], [batch_size, units]]
        x = self.fc(output)  # [batch_size, units]
        x = self.fc_vocab(x)  # [batch_size, vocab_size]
        return x, state, attention_weights

    def init_state(self, f_for_capt):
        return [self.fc_m0(f_for_capt), self.fc_c0(f_for_capt)]  # [batch_size, units]
