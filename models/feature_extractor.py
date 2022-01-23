import tensorflow as tf


def get_feature_extractor():
    image_model = tf.keras.applications.InceptionV3(include_top=True,
                                                    weights='imagenet')
    new_input = image_model.input
    f_for_attn_output = image_model.layers[-3].output
    f_for_capt_output = image_model.layers[-2].output
    image_features_extract_model = tf.keras.Model(new_input, [f_for_attn_output, f_for_capt_output])
    return image_features_extract_model
