import numpy as np


def circle(shape, pos, radius, value):

    # create meshgrid of coords
    xx, yy = np.mgrid[:shape[1], :shape[0]]

    # calc squared distance to pos
    circle = (xx - pos[1]) ** 2 + (yy - pos[0]) ** 2

    return (circle <= radius**2) * value