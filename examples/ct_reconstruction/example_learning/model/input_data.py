import tensorflow as tf
import numpy as np
from .geometry_parameters import GEOMETRY
from pyronn.ct_reconstruction.helpers.phantoms import circle
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d


def generate_training_data(number_of_training_samples_generate):

    train_data = np.empty((number_of_training_samples_generate,) + tuple(GEOMETRY.sinogram_shape))
    labels     = np.empty((number_of_training_samples_generate,) + tuple(GEOMETRY.volume_shape))

    with tf.Session() as sess:
        for i in range(number_of_training_samples_generate):
            # build random circle
            pos    = np.array([np.random.uniform(0, GEOMETRY.volume_shape[0]), np.random.uniform(0, GEOMETRY.volume_shape[1])], dtype=np.int32)
            radius = int(np.random.uniform(0, (np.sqrt(GEOMETRY.volume_shape[0]**2+GEOMETRY.volume_shape[1]**2))))/2
            value  = np.random.uniform(0.0, 1.0)
            labels[i] = circle.circle(GEOMETRY.volume_shape, pos, radius, value)

            # project it
            train_data[i] = generate_sinogram(labels[i], parallel_projection2d, GEOMETRY)

    return train_data, labels