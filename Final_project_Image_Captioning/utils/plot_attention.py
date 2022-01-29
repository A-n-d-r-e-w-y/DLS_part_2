from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_attention(image_path, result, attention_plot, height=15, weight=15):
    temp_image = np.array(Image.open(image_path))
    fig = plt.figure(figsize=(height, weight))
    fig.tight_layout()
    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result / 5)), 5)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.5, extent=img.get_extent())
    plt.show()
