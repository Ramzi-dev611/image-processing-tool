import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from TP1 import read_pgm


def otsu(matrix, height, width, maximum_grayscale):

    pixel_number = width * height
    flat_image = matrix.reshape(pixel_number, 1)
    mean_weigth = 1.0 / pixel_number

    his, bins = np.histogram(flat_image, np.array(range(0, maximum_grayscale + 1)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]:
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value

    print(final_thresh)
    final_image = flat_image.copy()
    final_image[flat_image > final_thresh] = 255
    final_image[flat_image < final_thresh] = 0
    new_image = final_image.reshape(height, width)

    plt.imshow(Image.fromarray(new_image))
    plt.show()


if __name__ == '__main__':
    li, lj, gray_scale, data = read_pgm('assets/balloons-P2.pgm')
    otsu(data, li, lj, gray_scale)
