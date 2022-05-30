import numpy as np
import matplotlib.pyplot as plt
from TP1 import read_pgm, write_pgm
from PIL import Image


def distort_image(matrix, height, width, max_grayscale, save_option= True):
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
    response = response.reshape(height, width)
    if save_option:
        write_pgm('distorted-image', response, max_grayscale, height, width)
    return response


def filter_image(matrix, height, width, max_grayscale, applied_filter, filter_size, save_option=True):
    response = np.zeros(width*height, dtype=int)
    skipped = int((filter_size-1)/2)
    for i, row in enumerate(matrix):
        if i in range(skipped) or i in range(height - skipped, height):
            continue
        else:
            for j, col in enumerate(row):
                if j in range(skipped) or j in range(width - skipped, width):
                    continue
                else:
                    element = matrix[np.ix_(list(range(i-skipped, i+skipped+1)), list(range(j-skipped, j+skipped+1)))]
                    response[i*width+j] = convolution_three_by_three(element, applied_filter) \
                        if convolution_three_by_three(element, applied_filter) > 0 else 0
    response = response.reshape(height, width)
    plt.imshow(Image.fromarray(response))
    plt.show()
    if save_option:
        write_pgm('convolution-image', response, max_grayscale, height, width)
    return response


def convolution_three_by_three(a, b):
    convolution = 0
    for i in range(3):
        for j in range(3):
            convolution += a[i][j] * b[i][j]
    return convolution


def median_filter_image(matrix, height, width, max_grayscale, filter_size, save_option=True):
    response = np.zeros(width * height, dtype=int)
    skipped = int((filter_size - 1) / 2)
    for i, row in enumerate(matrix):
        if i in range(skipped) or i in range(height - skipped, height):
            continue
        else:
            for j, col in enumerate(row):
                if j in range(skipped) or j in range(width - skipped, width):
                    continue
                else:
                    element = matrix[np.ix_(list(range(i - skipped, i + skipped + 1)), list(range(j - skipped, j + skipped + 1)))]
                    element = element.reshape(filter_size**2, 1)
                    response[i * width + j] = np.sort(element)[int((filter_size**2+1)/2)]
    response = response.reshape(height, width)
    if save_option:
        write_pgm('median-image', response, max_grayscale, height, width)
    return response


# definition of the low pass filter mean of size 3X3
low_pass_filter_three = (1/9) * np.ones(9).reshape(3, 3)

# definition of the low pass filter mean of size 5X5
low_pass_filter_five = (1/25) * np.ones(25).reshape(5, 5)

# definition of the low pass filter mean of size 7X7
low_pass_filter_seven = (1/9) * np.ones(49).reshape(7, 7)

# definition of high pass filter of size 3X3
high_pass_filter_v1 = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape(3, 3)

# definition of high pass filter of size 3X3
high_pass_filter_v2 = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape(3, 3)

# definition of high pass filter of size 3X3
high_pass_filter_v3 = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape(3, 3)


if __name__ == '__main__':

    li, lj, gray_scale, data = read_pgm('./assets/equalized-image.pgm')
    distorted_image = distort_image(data, li, lj, gray_scale)
    # plot the distorted image
    plt.imshow(Image.fromarray(distorted_image))
    plt.show()

    filtered_low = filter_image(distorted_image, li, lj, gray_scale, low_pass_filter_three, 3)
    # plot low filtered image
    plt.imshow(Image.fromarray(filtered_low))
    plt.show()

    filtered_high = filter_image(distorted_image, li, lj, gray_scale, high_pass_filter_v1, 3)
    # plot high filtered image
    plt.imshow(Image.fromarray(filtered_high))
    plt.show()

    median_filtered_image = median_filter_image(distorted_image, li, lj, gray_scale, 3)
    plt.imshow(Image.fromarray(median_filtered_image))
    plt.show()

