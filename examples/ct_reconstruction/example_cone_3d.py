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

from pyronn.ct_reconstruction.layers.projection_3d import cone_projection3d
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak_3D


def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size, volume_size]
    volume_spacing = [0.5, 0.5, 0.5]

    # Detector Parameters:
    detector_shape = [2*volume_size, 2*volume_size]
    detector_spacing = [1, 1]

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2 * np.pi

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_3d(geometry))

    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)

    # ------------------ Call Layers ------------------
    # The following code is the new TF2.0 experimental way to tell
    # Tensorflow only to allocate GPU memory needed rather then allocate every GPU memory available.
    # This is important for the use of the hardware interpolation projector, otherwise there might be not enough memory left
    # to allocate the texture memory on the GPU

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RunetimeError as e:
            print(e)

    sinogram = cone_projection3d(phantom, geometry)

    reco_filter = ram_lak_3D(geometry)
    sino_freq = tf.signal.fft(tf.cast(sinogram,dtype=tf.complex64))
    sino_filtered_freq = tf.multiply(sino_freq,tf.cast(reco_filter,dtype=tf.complex64))
    sinogram_filtered = tf.math.real(tf.signal.ifft(sino_filtered_freq))

    reco = cone_backprojection3d(sinogram_filtered, geometry)

    plt.figure()
    plt.imshow(np.squeeze(reco)[volume_shape[0]//2], cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig('3d_cone_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_cone_3d()
