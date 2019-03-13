import numpy as np
import tensorflow as tf
from .geometry_parameters import GEOMETRY
from pyronn.ct_reconstruction.helpers.phantoms import primitives_2d, shepp_logan
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d


def generate_training_data(number_of_samples, noise_sample_percentage=0.1):

    number_of_noise_samples = int(number_of_samples * noise_sample_percentage)

    data   = np.empty((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
    labels = np.empty((number_of_samples,) + tuple(GEOMETRY.volume_shape))

    with tf.Session() as sess:

        # build growing circles
        for i in range(number_of_samples-number_of_noise_samples):
            pos    = GEOMETRY.volume_shape//2 # middle
            radius = 5 + i
            value  = 1.0 # np.random.uniform(0.01, 1.0)
            labels[i] = primitives_2d.circle(GEOMETRY.volume_shape, pos, radius, value)

            # project it
            data[i] = generate_sinogram(labels[i], parallel_projection2d, GEOMETRY)

        # add some noise only phantoms
        for i in range(number_of_samples-number_of_noise_samples, number_of_samples):
            labels[i] = np.random.uniform(0.0, 1.0, GEOMETRY.volume_shape)
            data[i] = generate_sinogram(labels[i], parallel_projection2d, GEOMETRY)

    return data, labels


def generate_validation_data(number_of_samples):

    data   = np.empty((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
    labels = np.empty((number_of_samples,) + tuple(GEOMETRY.volume_shape))

    with tf.Session() as sess:

        # build growing rectangles
        for i in range(number_of_samples):
            size   = np.array([5, 5]) + 10*i
            pos    = GEOMETRY.volume_shape//2 - size//2 # middle
            value  = np.random.uniform(0.01, 1.0)
            labels[i] = primitives_2d.rect(GEOMETRY.volume_shape, pos, size, value)

            # project it
            data[i] = generate_sinogram(labels[i], parallel_projection2d, GEOMETRY)

    return data, labels


def get_test_data(number_of_samples=1):

    data = np.empty((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
    labels = np.empty((number_of_samples,) + tuple(GEOMETRY.volume_shape))

    # get shepp logan 2d
    if number_of_samples == 1:
         labels[0] = shepp_logan.shepp_logan_enhanced(GEOMETRY.volume_shape)
         data[0] = generate_sinogram(labels[0], parallel_projection2d, GEOMETRY)

    # every slice of shepp logan 3d with number_of_samples as Z-dimension as own image
    else:
        labels = shepp_logan.shepp_logan_3d((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
        for i in range(number_of_samples):
            data[i] = generate_sinogram(labels[i], parallel_projection2d, GEOMETRY)

    return data, labels

