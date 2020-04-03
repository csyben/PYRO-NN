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

        self.cosine_weight = tf.Variable(name='cosine_weight', dtype=tf.float32,
                                             initial_value=ct_weights.cosine_weights_3d(self.geometry), trainable=False)

        self.redundancy_weight = tf.Variable(name='redundancy_weight', dtype=tf.float32,
                                             initial_value=ct_weights.parker_weights_3d(self.geometry), trainable=False)

        self.filter = tf.Variable(name='reco_filter', dtype=tf.float32, initial_value=ram_lak_3D(self.geometry), trainable=False)



    def model(self, sinogram):
        self.sinogram_cos = tf.multiply(sinogram, self.cosine_weight)
        self.redundancy_weighted_sino = tf.multiply(self.sinogram_cos,self.redundancy_weight)

        self.weighted_sino_fft = tf.signal.fft(tf.cast(self.redundancy_weighted_sino, dtype=tf.complex64))
        self.filtered_sinogram_fft = tf.multiply(self.weighted_sino_fft, tf.cast(self.filter,dtype=tf.complex64))
        self.filtered_sinogram = tf.math.real(tf.signal.ifft(self.filtered_sinogram_fft))

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

    geometry.set_trajectory(projection_geometry)

    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    phantom = np.expand_dims(phantom,axis=0)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RunetimeError as e:
    #         print(e)
    # ------------------ Call Layers ------------------

    sinogram = generate_sinogram.generate_sinogram(phantom, cone_projection3d, geometry)

    model = nn_model(geometry)
    reco, redundancy_weighted_sino = model.model(sinogram)

    plt.figure()
    plt.imshow(np.squeeze(reco)[volume_shape[0]//2], cmap=plt.get_cmap('gist_gray'), vmin=0, vmax=0.4)
    plt.axis('off')
    plt.savefig('fdk_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_cone_3d()
