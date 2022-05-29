import numpy as np
import matplotlib.pyplot as plt
from TP1 import read_pgm, write_pgm
from PIL import Image


def distort_image(matrix, height, width, max_grayscale):
    np.random.seed(0)
    response = np.zeros(width * height, dtype=int)
    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            distortion_factor = np.random.randint(0, 20)
            if distortion_factor == 0:
                response[width * i+j] =0
            elif distortion_factor == 20:
                response[width * i+j] = max_grayscale
            else:
                response[width * i+j] = col
    return response.reshape(height, width)


def filter_image(matrix, height, width, applied_filter):
    response = np.zeros(width*height, dtype=int)
    for i, row in enumerate(matrix):
        if i == 0 or i == height-1:
            continue
        else:
            for j, col in enumerate(row):
                if j == 0 or j == width -1:
                    pass
                else:
                    element = np.array([
                        matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i - 1][j + 1],
                        matrix[i][j - 1], col, matrix[i][j + 1],
                        matrix[i + 1][j - 1], matrix[i + 1][j], matrix[i + 1][j + 1]
                    ]).reshape(3, 3)
                    response[i*width+j] = convolution_three_by_three(element, applied_filter) if convolution_three_by_three(element, applied_filter) > 0 else 0
    return response.reshape(height, width)


def convolution_three_by_three(a, b):
    convolution = 0
    for i in range(3):
        for j in range(3):
            convolution += a[i][j] * b[i][j]
    return convolution


if __name__ == '__main__':
    li, lj, gray_scale, data = read_pgm('./assets/equalized-image.pgm')
    distorted_image = distort_image(data, li, lj, gray_scale)
    # plot the distorted image
    # plt.imshow(Image.fromarray(distorted_image))
    # plt.show()

    # definition of the low pass filter mean of size 3X3
    low_pass_filter = (1/3) * np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(3,3)

    # definition of high pass filter of size 3X3
    high_pass_filter = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape(3, 3)

    filtered_low = filter_image(distorted_image, li, lj, low_pass_filter)
    plt.imshow(Image.fromarray(filtered_low))
    plt.show()

    filtered_high = filter_image(distorted_image, li, lj, high_pass_filter)
    plt.imshow(Image.fromarray(filtered_high))
    plt.show()
