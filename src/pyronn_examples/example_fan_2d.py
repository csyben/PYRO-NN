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
import matplotlib.pyplot as plt

# TODO: better imports
from pyronn.ct_reconstruction.layers.projection_2d import FanProjectionFor2D
from pyronn.ct_reconstruction.layers.backprojection_2d import FanBackProjectionFor2D
from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.filters import weights
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.misc.general_utils import fft_and_ifft
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d


def example_fan_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 512
    volume_shape = [volume_size, volume_size]
    volume_spacing = [0.23655,0.23655]

    # Detector Parameters:
    detector_shape = [512]
    detector_spacing = [0.8, 0.8]

    # Trajectory Parameters:
    number_of_projections = 512
    angular_range = 2*np.pi

    sdd = 200
    sid = 100

    # create Geometry class
    geometry = GeometryFan2D(volume_shape, volume_spacing,
                             detector_shape, detector_spacing,
                             number_of_projections, angular_range,
                             sdd, sid)
    geometry.set_trajectory(circular_trajectory_2d(geometry.number_of_projections, geometry.angular_range, True))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    # Add required batch dimension
    phantom = np.expand_dims(phantom,axis=0)
    # ------------------ Call Layers ------------------

    sinogram = FanProjectionFor2D().forward(phantom, geometry)

    #TODO: Add Cosine weighting

    redundancy_weights = weights.parker_weights_2d(geometry)
    sinogram_redun_weighted = sinogram * redundancy_weights
    reco_filter = filters.ram_lak_2D(detector_shape, detector_spacing, number_of_projections)
    x = fft_and_ifft(sinogram, reco_filter)

    reco = FanBackProjectionFor2D().forward(x, geometry)

    plt.figure()
    plt.imshow(np.squeeze(reco), cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig(f'2d_fan_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_fan_2d()
