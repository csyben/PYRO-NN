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
import matplotlib.pyplot as plt

from pyronn.ct_reconstruction.helpers.misc import generate_sinogram
from pyronn.ct_reconstruction.layers.projection_3d import cone_projection3d
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak_3D
import pyronn.ct_reconstruction.helpers.filters.weights as ct_weights



class nn_model:
    def __init__(self, geometry):
        self.geometry = geometry

        self.cosine_weight = tf.get_variable(name='cosine_weight', dtype=tf.float32,
                                             initializer=ct_weights.cosine_weights_3d(self.geometry), trainable=False)

        self.redundancy_weight = tf.get_variable(name='redundancy_weight', dtype=tf.float32,
                                             initializer=ct_weights.parker_weights_3d(self.geometry), trainable=False)

        self.filter = tf.get_variable(name='reco_filter', dtype=tf.float32, initializer=ram_lak_3D(self.geometry), trainable=False)



    def model(self, sinogram):
        self.sinogram_cos = tf.multiply(sinogram, self.cosine_weight)
        self.redundancy_weighted_sino = tf.multiply(self.sinogram_cos,self.redundancy_weight)

        self.weighted_sino_fft = tf.fft(tf.cast(self.redundancy_weighted_sino, dtype=tf.complex64))
        self.filtered_sinogram_fft = tf.multiply(self.weighted_sino_fft, tf.cast(self.filter,dtype=tf.complex64))
        self.filtered_sinogram = tf.real(tf.ifft(self.filtered_sinogram_fft))

        self.reconstruction = cone_backprojection3d(self.filtered_sinogram,self.geometry, hardware_interp=True)

        return self.reconstruction, self.redundancy_weighted_sino


def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size, volume_size]
    v_spacing = 0.25
    volume_spacing = [v_spacing,v_spacing,v_spacing]

    # Detector Parameters:
    detector_shape = [450 , 450]
    d_spacing = 0.33
    detector_spacing = [d_spacing,d_spacing]

    # Trajectory Parameters:
    number_of_projections = 248
    angular_range = np.pi+2*np.arctan(detector_shape[0] / 2 / 1200)

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.angular_range = np.radians(200)
    projection_geometry = circular_trajectory.circular_trajectory_3d(geometry)

    geometry.set_projection_matrices(projection_geometry)

    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)



    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config=config) as sess:
        sinogram = generate_sinogram.generate_sinogram(phantom, cone_projection3d, geometry)

        model = nn_model(geometry)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        reco_tf, redundancy_weighted_sino_tf = model.model(sinogram)
        reco, redundancy_weighted_sino = sess.run([reco_tf, redundancy_weighted_sino_tf])

    plt.figure()
    plt.imshow(reco[(int)(volume_shape[0]/2),:,:], cmap=plt.get_cmap('gist_gray'), vmin=0, vmax=0.4)
    plt.axis('off')
    plt.savefig('fdk_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_cone_3d()
