import math
import matplotlib.pyplot as plt

import numpy as np


def create_matrix_from_list(li, lj, content):
    data = np.asarray(content, dtype=int)
    reshaped = data.reshape(li, lj)
    return reshaped


def read_pgm(path: str):
    image_file = None
    image_type = ""
    try:
        image_file = open(path, "r")
        image_type = image_file.readline()
    except UnicodeDecodeError:
        image_file = open(path, "rb")
        image_type = image_file.readline().decode()
    finally:
        if image_type == 'P2\n':
            return get_image_content_type_2(image_file)
        elif type == 'P5\n':
            get_image_content_type_5(image_file)
        else:
            raise Exception('something is wrong with the provided file')
        return


def get_image_content_type_2(file):
    # get lines contained in the image
    lines = file.readlines()
    # remove comment
    without_comments = [line for line in lines if line[0] != '#']
    # get nb lines li and nb columns lj of the image
    [lj, li] = [int(c) for c in without_comments[0].split(" ")]
    # get maximum gray level
    gray_scale = int(without_comments[1])
    # get the image content
    content = []
    for line in without_comments[2:]:
        content.extend([int(c) for c in line.split()])
    data = create_matrix_from_list(li, lj, content)
    return li, lj, gray_scale, data


def get_image_content_type_5(file):
    print("I am not going to read a binary image")


def write_pgm(name, data, gray_scale, li, lj):
    file = open(f"assets/{name}.pgm", "w")
    file.write("P2\n")
    file.write(f"{lj} {li}\n")
    file.write(f"{gray_scale}\n")
    for row in data:
        line = ''
        for col in row:
            line += str(col)+' '
        file.write(line)


def get_image_stats(data, li, lj):
    reshaped = data.reshape(li*lj, 1)
    mean = np.mean(reshaped)
    variance = np.var(reshaped)
    return mean, math.sqrt(variance)


def get_histogram(data, grayscale):
    histogram = np.zeros(grayscale+1, dtype=int)
    for row in data:
        for col in row:
            histogram[col] += 1
    plt.bar(range(grayscale+1), histogram)
    plt.show()


def get_cumulative_histogram(data, grayscale):
    response = np.zeros(grayscale+1, dtype=int)
    for row in data:
        for col in row:
            response[col] += 1
    for index in range(1, grayscale+1):
        response[index] += response[index-1]
    plt.bar(range(grayscale + 1), response)
    plt.show()


if __name__ == '__main__':
    # read image
    li, lj, gray_scale, data = read_pgm('./assets/balloons-P2.pgm')
    # save image
    write_pgm("new_image", data, gray_scale, li, lj)
    # get the mean and standard div
    get_image_stats(data, li, lj)
    # get the histogram
    get_histogram(data, gray_scale)
    # get the cumulative histogram
    get_cumulative_histogram(data, gray_scale)
