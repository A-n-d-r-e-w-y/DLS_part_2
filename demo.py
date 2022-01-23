import os
import pickle

import tensorflow as tf

from models import CONSTANTS
from models.encoder import CNN_Encoder
from models.decoder import RNN_Attn_Decoder
from models.feature_extractor import get_feature_extractor
from utils.greedy_search import greedy_search
from utils.beam_search import beam_search
from utils.sampling_search import sampling_search


def initialization():
    os.system("unzip -qqq -u pretrained_data/lstm/encoder_weights.zip")
    os.system("unzip -qqq -u pretrained_data/lstm/decoder_weights.zip")
    with open('pretrained_data/lstm/vocabulary.pkl', 'rb') as handle:
        vocabulary = pickle.load(handle)
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary)
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary,
        invert=True)

    image_features_extract_model = get_feature_extractor()

    encoder = CNN_Encoder(CONSTANTS.UNITS)
    decoder = RNN_Attn_Decoder(CONSTANTS.EMBEDDING_DIM, CONSTANTS.UNITS, len(vocabulary))
    encoder.load_weights("encoder_weights/")
    decoder.load_weights("decoder_weights/")

    def greedy_search_demo(image_path):
        return greedy_search(image_path=image_path,
                             image_features_extract_model=image_features_extract_model,
                             encoder=encoder,
                             decoder=decoder,
                             word_to_index=word_to_index,
                             index_to_word=index_to_word,
                             attention_features_shape=CONSTANTS.ATTENTION_FEATURES_SHAPE)

    def beam_search_demo(image_path, top_k):
        return beam_search(image_path=image_path,
                           image_features_extract_model=image_features_extract_model,
                           encoder=encoder,
                           decoder=decoder,
                           word_to_index=word_to_index,
                           index_to_word=index_to_word,
                           top_k=top_k)

    def sampling_search_demo(image_path):
        return sampling_search(image_path=image_path,
                               image_features_extract_model=image_features_extract_model,
                               encoder=encoder,
                               decoder=decoder,
                               word_to_index=word_to_index,
                               index_to_word=index_to_word,
                               attention_features_shape=CONSTANTS.ATTENTION_FEATURES_SHAPE)

    return greedy_search_demo, beam_search_demo, sampling_search_demo
