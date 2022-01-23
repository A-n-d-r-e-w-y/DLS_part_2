import numpy as np
import tensorflow as tf
from utils.modules import load_image


def beam_search(image_path,
                image_features_extract_model,
                encoder,
                decoder,
                word_to_index,
                index_to_word,
                max_length=40,
                top_k=3):
    temp_input = tf.expand_dims(load_image(image_path), 0)
    f_for_attn, f_for_capt = image_features_extract_model(temp_input)
    img_tensor = tf.reshape(f_for_attn, (f_for_attn.shape[0], -1, f_for_attn.shape[3]))
    features = encoder(img_tensor)
    hidden = decoder.init_state(f_for_capt)
    BOS_IDX = tf.convert_to_tensor('<bos>')
    EOS_IDX = tf.convert_to_tensor('<eos>')
    dec_input = tf.convert_to_tensor([word_to_index(BOS_IDX)])
    results = [(0, dec_input, hidden, ['<bos>'])]  # proba_sum, hypothesis, last_hidden, words
    for i in range(max_length):
        new_results = []
        for result in results:
            hypothesis = result[1]
            if hypothesis[-1] == word_to_index(EOS_IDX):
                new_results.append(result)
            else:
                dec_input = tf.convert_to_tensor([hypothesis[-1]])
                hidden = result[2]
                predictions, hidden, _ = decoder(dec_input, features, hidden)
                probas = tf.nn.softmax(predictions, -1)[0].numpy()
                top_idx = tf.math.top_k(predictions, k=top_k).indices.numpy()[0]
                for top_i in top_idx:
                    predicted_word = tf.compat.as_text(index_to_word(tf.convert_to_tensor(top_i)).numpy())
                    top_i_expand = tf.cast([top_i], dtype=tf.int64)
                    new_results.append((result[0] + np.log(probas[top_i]),
                                        tf.concat([hypothesis, top_i_expand], axis=0),
                                        hidden,
                                        result[3] + [predicted_word]))
        results = sorted(new_results, reverse=True)[:top_k]
    return results
