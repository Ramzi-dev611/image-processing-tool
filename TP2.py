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
    return new_data


def histogram_linear_transformation(matrix, level0, level1, maximum_grayscale, plot_option=False, save_option=True):
    height, width = matrix.shape
    response = np.zeros(height*width, dtype=int)
    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            if col <= level0:
                response[width*i+j] += 0
            elif col >= level1:
                response[width*i+j] += maximum_grayscale
            else:
                response[width*i+j] += maximum_grayscale * (col-level0) /(level1 - level0)
    response = response.reshape(height, width)
    if plot_option:
        histogram = get_histogram(response, grayscale=maximum_grayscale)
        plt.bar(range(maximum_grayscale+1), histogram)
        plt.show()
    if save_option:
        write_pgm('linear-transformation', response, maximum_grayscale, height, width)
    return response


if __name__ == "__main__":
    li, lj, gray_scale, data = read_pgm('assets/balloons-P2.pgm')
    equalize_histogram(data, li, lj, gray_scale, plot_option=True)
    histogram_linear_transformation(data, 50, 150, gray_scale, plot_option=True)