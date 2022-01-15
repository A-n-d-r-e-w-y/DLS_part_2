from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_attention(image_path, result, attention_plot, height=10, weight=10):
    temp_image = np.array(Image.open(image_path))
    fig = plt.figure(figsize=(height, weight))
    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = min(4, len_result)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.5, extent=img.get_extent())
    plt.tight_layout()
    plt.show()
