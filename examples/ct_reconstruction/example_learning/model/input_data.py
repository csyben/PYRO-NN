# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from .geometry_parameters import GEOMETRY
from pyronn.ct_reconstruction.helpers.phantoms import primitives_2d, shepp_logan
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram_parallel_2d


def generate_training_data(number_of_samples, num_noise_sample_percentage=0.0):

    number_of_noise_samples = int(number_of_samples * num_noise_sample_percentage)
    number_of_circles = number_of_samples - number_of_noise_samples
    growth_rate = np.min(GEOMETRY.volume_shape) / number_of_circles / 2

    data   = np.empty((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
    labels = np.empty((number_of_samples,) + tuple(GEOMETRY.volume_shape))

    with tf.Session() as sess:

        # build growing circles until volume is filled
        for i in range(number_of_circles):
            pos    = GEOMETRY.volume_shape//2 # middle
            radius = int(growth_rate * (i+1))
            value  = 1.0 #np.random.uniform(0.01, 1.0) # here one could also try random values for each circle
            labels[i] = primitives_2d.circle(GEOMETRY.volume_shape, pos, radius, value)

            # project it
            data[i] = generate_sinogram_parallel_2d(labels[i], GEOMETRY)

        # add some noise only phantoms
        for i in range(number_of_samples-number_of_noise_samples, number_of_samples):
            labels[i] = np.random.uniform(0.0, 1.0, GEOMETRY.volume_shape)
            data[i] = generate_sinogram_parallel_2d(labels[i], GEOMETRY)

    return data, labels


def generate_validation_data(number_of_samples):

    growth_rate = np.min(GEOMETRY.volume_shape) / number_of_samples / 2

    data   = np.empty((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
    labels = np.empty((number_of_samples,) + tuple(GEOMETRY.volume_shape))

    with tf.Session() as sess:

        # build growing rectangles
        for i in range(number_of_samples):
            size   = np.array([growth_rate * (i+1), growth_rate * (i+1)], dtype=np.int32)
            pos    = GEOMETRY.volume_shape//2 - size//2 # middle
            value  = np.random.uniform(0.01, 1.0)
            labels[i] = primitives_2d.rect(GEOMETRY.volume_shape, pos, size, value)

            # project it
            data[i] = generate_sinogram_parallel_2d(labels[i], GEOMETRY)

    return data, labels


def get_test_data(number_of_samples=1):

    data = np.empty((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
    labels = np.empty((number_of_samples,) + tuple(GEOMETRY.volume_shape))

    with tf.Session() as sess:

        # get shepp logan 2d
        if number_of_samples == 1:
             labels[0] = shepp_logan.shepp_logan_enhanced(GEOMETRY.volume_shape)
             data[0] = generate_sinogram_parallel_2d(labels[0], GEOMETRY)

        # every slice of shepp logan 3d with number_of_samples as Z-dimension as own image
        else:
            labels = shepp_logan.shepp_logan_3d((number_of_samples,) + tuple(GEOMETRY.sinogram_shape))
            for i in range(number_of_samples):
                data[i] = generate_sinogram_parallel_2d(labels[i], GEOMETRY)

    return data, labels


def get_test_cupping_data():

    data = np.empty((1,) + tuple(GEOMETRY.sinogram_shape))
    labels = np.empty((1,) + tuple(GEOMETRY.volume_shape))

    with tf.Session() as sess:
         labels[0] = primitives_2d.circle(GEOMETRY.volume_shape,
                                          GEOMETRY.volume_shape//2,
                                          np.min(GEOMETRY.volume_shape//2))
         data[0] = generate_sinogram_parallel_2d(labels[0], GEOMETRY)


    return data, labels
