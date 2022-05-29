import math

from TP1 import read_pgm, write_pgm, get_histogram, get_cumulative_histogram
import numpy as np
import matplotlib.pyplot as plt


def equalize_histogram(matrix, height, width, maximum_grayscale, plot_option=False, save_option=True):
    hc = get_cumulative_histogram(matrix, maximum_grayscale)
    cumulative_probability = [float(c/(width*height)) for c in hc]
    grayscale_mapping = [int(math.ceil(c * maximum_grayscale)) for c in cumulative_probability]
    new_data = np.zeros(width*height, dtype=int)
    for i in range(height):
        for j in range(width):
            new_data[i*width+j] = grayscale_mapping[data[i][j]]
    new_data = new_data.reshape(height, width)
    if plot_option:
        new_histogram = get_histogram(new_data, maximum_grayscale)
        plt.bar(range(maximum_grayscale+1), new_histogram)
        plt.show()
    if save_option:
        write_pgm('equalized-image', new_data, maximum_grayscale, height, width)


if __name__ == "__main__":
    li, lj, gray_scale, data = read_pgm('assets/balloons-P2.pgm')
    equalize_histogram(data, li, lj, gray_scale, plot_option=True)