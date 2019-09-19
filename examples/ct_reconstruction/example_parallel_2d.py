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

from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory


def example_parallel_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = 800
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2* np.pi

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    phantom = np.expand_dims(phantom, axis=0)

    # ------------------ Call Layers ------------------
    with tf.compat.v1.Session() as sess:
        result = parallel_projection2d(phantom, geometry)
        sinogram = result.eval()

        #sinogram = sinogram + np.random.normal(
        #    loc=np.mean(np.abs(sinogram)), scale=np.std(sinogram), size=sinogram.shape) * 0.02

        reco_filter = filters.ram_lak_2D(geometry)

        sino_freq = np.fft.fft(sinogram, axis=1)
        sino_filtered_freq = np.multiply(sino_freq,reco_filter)
        sinogram_filtered = np.fft.ifft(sino_filtered_freq, axis=1)

        result_back_proj = parallel_backprojection2d(sinogram_filtered, geometry)
        reco = result_back_proj.eval()

        plt.figure()
        plt.imshow(np.squeeze(reco), cmap=plt.get_cmap('gist_gray'))
        plt.axis('off')
        plt.savefig('2d_par_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_parallel_2d()
