import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from TP1 import read_pgm
from TP3 import filter_image, low_pass_filter_three, distort_image


def otsu(matrix, height, width, maximum_grayscale, fixed_thresh_hold=None):

    pixel_number = width * height
    flat_image = matrix.reshape(pixel_number, 1)
    mean_weigth = 1.0 / pixel_number

    his, bins = np.histogram(flat_image, np.array(range(0, maximum_grayscale + 1)))
    final_thresh = -1
    final_value = -1
    if fixed_thresh_hold is None:
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
    else:
        final_thresh = fixed_thresh_hold
    final_image = flat_image.copy()
    final_image[flat_image > final_thresh] = 0
    final_image[flat_image < final_thresh] = 255
    new_image = final_image.reshape(height, width)

    plt.imshow(Image.fromarray(new_image))
    plt.show()


def get_object_borders(matrix, height, width, maximum_grayscale, applied_filter):
    filtered_image = filter_image(matrix, height, width, maximum_grayscale, applied_filter, len(applied_filter), save_option=False)
    response = matrix - filtered_image
    response[response != 0] = 255
    plt.imshow(Image.fromarray(response))
    plt.show()
    return response


def dilatation(matrix, width, height, object):
    response = 255 * np.zeros(width * height)
    skipped = int((len(object)-1)/2)
    for i, row in enumerate(matrix):
        if i in range(skipped) or i in range(height - skipped, height):
            continue
        else:
            for j, col in enumerate(row):
                if j in range(skipped) or j in range(width - skipped, width):
                    if col == 0:
                        for line in range(2 * skipped +1):
                            for column in range(2 * skipped +1):
                                response[(i-skipped+line) * width + j-skipped+column] = 0
    response = response.reshape(height, width)
    plt.imshow(Image.fromarray(response))
    plt.show()
    return response


def erosion(matrix, width, height, max_grayscale, object):
    pass


def ouverture(matrix, width, height, max_grayscale, object):
    pass


def fermeture(matrix, width, height, max_grayscale, object):
    pass


square_three = 255*np.ones(9).reshape(3, 3)
square_five = 255*np.ones(25).reshape(5, 5)
square_seven = 255*np.ones(49).reshape(7, 7)



if __name__ == '__main__':
    li, lj, gray_scale, data = read_pgm('assets/balloons-P2.pgm')
    otsu(data, li, lj, gray_scale)
    # distorted_image = distort_image(data, li, lj, gray_scale, save_option=False)
    # binary = get_object_borders(distorted_image, li, lj, gray_scale, low_pass_filter_three)
    # dilated = dilatation(binary, li, lj, square_three)
