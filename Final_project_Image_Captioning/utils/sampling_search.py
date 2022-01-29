import numpy as np
import tensorflow as tf
from utils.modules import load_image


def sampling_search(image_path,
                    image_features_extract_model,
                    encoder,
                    decoder,
                    word_to_index,
                    index_to_word,
                    attention_features_shape=64,
                    max_length=40):
    attention_plot = np.zeros((max_length, attention_features_shape))
    temp_input = tf.expand_dims(load_image(image_path), 0)
    f_for_attn, f_for_capt = image_features_extract_model(temp_input)
    img_tensor = tf.reshape(f_for_attn, (f_for_attn.shape[0], -1, f_for_attn.shape[3]))
    features = encoder(img_tensor)
    state = decoder.init_state(f_for_capt)
    BOS_IDX = tf.convert_to_tensor('<bos>')
    dec_input = tf.convert_to_tensor([word_to_index(BOS_IDX)])
    result = []
    for i in range(max_length):
        predictions, state, attention_weights = decoder(dec_input,
                                                        features,
                                                        state)
        if attention_weights is not None:
            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(tf.convert_to_tensor(predicted_id)).numpy())
        result.append(predicted_word)
        if predicted_word == '<eos>':
            return result, attention_plot
        dec_input = tf.convert_to_tensor([predicted_id])
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot
