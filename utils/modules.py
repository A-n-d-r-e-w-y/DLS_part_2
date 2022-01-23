import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def show_image(image_path, height=10, weight=10):
    img = Image.open(image_path)
    plt.figure(figsize=(height, weight))
    plt.imshow(img)
    plt.show()


def post_processing(text):
    words_to_remove = ['<bos>', '<eos>', '[UNK]']
    for word in words_to_remove:
        text = text.replace(word, "")
    return text
