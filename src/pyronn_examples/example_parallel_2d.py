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

from pyronn.ct_reconstruction.layers.projection_2d import ParallelProjectionFor2D
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjectionFor2D
from pyronn.ct_reconstruction.geometry.geometry_base import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.misc.general_utils import fft_and_ifft
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d
import matplotlib.pyplot as plt

def example_parallel_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = [800]
    detector_spacing = [1]

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2* np.pi

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_trajectory(circular_trajectory_2d(geometry.number_of_projections, geometry.angular_range, True))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)

    # ------------------ Call Layers ------------------
    sinogram = ParallelProjectionFor2D().forward(phantom, geometry)

    #sinogram = sinogram + np.random.normal(
    #    loc=np.mean(np.abs(sinogram)), scale=np.std(sinogram), size=sinogram.shape) * 0.02

    reco_filter = filters.shepp_logan_2D(geometry.detector_shape, geometry.detector_spacing, geometry.number_of_projections)

    # # one for all
    # x = fft_and_ifft(sinogram, reco_filter)

    # You can also do it step by step
    import torch
    x = torch.fft.fft(torch.tensor(sinogram).cuda(), dim=-1, norm='ortho')
    x = torch.multiply(x, torch.tensor(reco_filter).cuda())
    x = torch.fft.ifft(x, dim=-1, norm='ortho').real

    reco = ParallelBackProjectionFor2D().forward(x, geometry)

    plt.figure()
    plt.imshow(np.squeeze(reco), cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig(f'2d_par_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_parallel_2d()
