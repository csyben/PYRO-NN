import numpy as np
from math import pow, sqrt, cos, sin, pi
import tensorflow as tf

# TODO: use my filter inits

def ramp(detector_width):
    filter_array = np.zeros(detector_width)
    frequency_spacing = 0.5 / (detector_width/2.0)
    for i in range(0, filter_array.shape[0]):
        if i <= filter_array.shape[0] / 2.0:
            filter_array[i] = i * frequency_spacing
        elif i > filter_array.shape[0] / 2.0:
            filter_array[i] = 0.5 - (((i - filter_array.shape[0]/2.0))*frequency_spacing)
    return filter_array.astype(np.float32)

def ram_lak(detector_width, detector_spacing):
    # TODO: Roll scaling to 0.25
    filter_array = np.zeros(detector_width)
    filter_array[0] = (0.25 / (detector_spacing * detector_spacing)) * 0.25
    odd = (-1.0 / (pi * pi * detector_spacing * detector_spacing)) * 0.25

    for i in range(1, int(filter_array.shape[0])):
        if i < filter_array.shape[0] / 2:
            if (i % 2) == 1:
                filter_array[i] = odd / (i * i)
        if i >= filter_array.shape[0] / 2:
            tmp = filter_array.shape[0] - i
            if (tmp % 2) == 1:
                filter_array[i] = odd / (tmp * tmp)

    filter_array = np.fft.fft(filter_array)
    return np.real(filter_array).astype(np.float32)

def ramp_3D(geometry):
    detector_width = geometry.detector_shape[np.alen(geometry.detector_shape) - 1]

    filter = [
        np.reshape(
            ramp(detector_width),
            (1, 1, detector_width)
        )
        for i in range(0, geometry.number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter

def ram_lak_3D(geometry):
    detector_width = geometry.detector_shape[np.alen(geometry.detector_shape) - 1]
    detector_spacing_width = geometry.detector_spacing[np.alen(geometry.detector_spacing) - 1]

    filter = [
        np.reshape(
            ram_lak(detector_width, detector_spacing_width),
            (1, 1, detector_width)
        )
        for i in range(0, geometry.number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter
