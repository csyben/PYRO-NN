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
import pyronn_layers

from pyronn.ct_reconstruction.geometry.geometry_parallel_3d import GeometryParallel3D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.layers.projection_3d import par_projection3d
from pyronn.ct_reconstruction.layers.backprojection_3d import par_backprojection3d


def circular_trajectory_3d(geometry):
    """
        Generates the central ray vectors defining a circular trajectory for use with the 2d projection layers.
    Args:
        geometry: 2d Geometry class including angular_range and number_of_projections
    Returns:
        Central ray vectors as np.array.
    """
    rays = np.zeros([geometry.number_of_projections, 3])
    angular_increment = geometry.angular_range / geometry.number_of_projections
    for i in range(geometry.number_of_projections):
        rays[i] = [np.cos(i * angular_increment), np.sin(i * angular_increment),0]
    return rays

def example_parallel_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size, volume_size]
    volume_spacing = [1, 1, 1]

    # Detector Parameters:
    detector_size = 350
    detector_shape= [detector_size, detector_size]
    detector_spacing = [1,1]

    # Trajectory Parameters:
    number_of_projections = 180
    angular_range = 1*np.pi

    # create Geometry class
    geometry = GeometryParallel3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_trajectory(circular_trajectory_3d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_3d(volume_shape)


    # ------------------ Call Layers ------------------
    with tf.Session() as sess:
        #result = parallel_projection3d(phantom, geometry)

        result = par_projection3d(phantom,geometry)

        sinogram = result.eval()




        sinogram = sinogram + np.random.normal(
           loc=np.mean(np.abs(sinogram)), scale=np.std(sinogram), size=sinogram.shape) * 0.02

        reco_filter = filters.ram_lak_3D(geometry)

        sino_freq = np.fft.fft(sinogram, axis=2)
        sino_filtered_freq = np.multiply(sino_freq,reco_filter)
        sinogram_filtered = np.fft.ifft(sino_filtered_freq, axis=2)

        result_back_proj = par_backprojection3d(sinogram_filtered,geometry)
        reco = result_back_proj.eval()
        import pyconrad as pyc
        pyc.setup_pyconrad()
        pyc.imshow(phantom)
        pyc.imshow(sinogram)
        pyc.imshow(reco)
        a = 5
        #plt.figure()
        #plt.imshow(reco, cmap=plt.get_cmap('gist_gray'))
        #plt.axis('off')
        #plt.savefig('3d_par_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_parallel_3d()
