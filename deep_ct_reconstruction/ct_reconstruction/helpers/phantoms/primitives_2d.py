import numpy as np


def circle(shape, pos, radius, value):

    # create meshgrid of coords
    xx, yy = np.mgrid[:shape[1], :shape[0]]

    # calc squared distance to pos
    circle = (xx - pos[1]) ** 2 + (yy - pos[0]) ** 2

    return (circle <= radius**2) * value

def ellipse(shape, pos, half_axes, value, theta=0):

    # create meshgrid of coords
    xx, yy = np.mgrid[:shape[0], :shape[1]]

    # move to pos
    xc = (xx - pos[1])
    yc = (yy - pos[0])

    # rotate
    xx = xc * np.cos(theta) + yc * np.sin(theta)
    yy = yc * np.cos(theta) - xc * np.sin(theta)

    a = half_axes[1]
    b = half_axes[0]

    # calc squared distance to pos
    ellipse_points = (xx ** 2) / (a**2) + (yy ** 2) / (b**2)

    return (ellipse_points <= 1) * value


def rect(shape, pos, size, value):

    # create array and populate it with value
    rectangle = np.zeros(shape)
    rectangle[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1]] = value

    return  rectangle
