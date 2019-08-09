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
    geometry.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geometry))

    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    phantom = np.expand_dims(phantom, axis=0)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config = config) as sess:
        result = cone_projection3d(phantom, geometry)
        sinogram = result.eval()

        #TODO: Use 3D ramp / ram_lak not 1D
        # filtering
        filter = ram_lak_3D(geometry)
        #filter = ramp(int(geometry.detector_shape[1]))
        sino_freq = np.fft.fft(sinogram, axis=-1)
        # filtered_sino_freq = np.zeros_like(sino_freq)
        # for row in range(int(geometry.detector_shape[0])):
        #     for projection in range(geometry.number_of_projections):
        #         filtered_sino_freq[projection, row, :] = sino_freq[projection, row, :] * filter[:]
        filtered_sino_freq = sino_freq * filter
        filtered_sino = np.fft.ifft(filtered_sino_freq, axis=-1)

        result_back_proj = cone_backprojection3d(filtered_sino, geometry)
        reco = result_back_proj.eval()
        plt.figure()
        plt.imshow(np.squeeze(reco)[volume_shape[0]//2], cmap=plt.get_cmap('gist_gray'))
        plt.axis('off')
        plt.savefig('3d_cone_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_cone_3d()
