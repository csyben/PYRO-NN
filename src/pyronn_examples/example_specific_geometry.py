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
from pyronn.ct_reconstruction.layers.projection_3d import ConeProjectionFor3D
from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjectionFor3D
from pyronn.ct_reconstruction.geometry.geometry_specific import SpecificGeometry
from pyronn.ct_reconstruction.geometry.geometry_base import GeometryCone3D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.filters import weights
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan, primitives_3d
from pyronn.ct_reconstruction.helpers.misc.general_utils import fft_and_ifft
from pyronn.ct_reconstruction.helpers.trajectories.arbitrary_trajectory import fibonacci_sphere_projecton_matrix
from pyronn.ct_reconstruction.helpers.filters.filters import shepp_logan_3D

class MySpecificGeometry(SpecificGeometry):
    def __init__(self, geo_dict_info):
        super().__init__(geo_dict_info, fibonacci_sphere_projecton_matrix)

    def set_geo(self):
        geometry = GeometryCone3D(**self.geometry_info)
        return geometry

    def create_sinogram(self, phantom):
        return ConeProjectionFor3D().forward(phantom, self.geometry)

def example_specific_geometry():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    geo_info = {
        'volume_size': 256,
        'volume_shape': [256, 256, 256],
        'volume_spacing': [0.5, 0.5, 0.5],

        'detector_shape': [400, 600],
        'detector_spacing': [1, 1],

        'number_of_projections': 360,
        'angular_range': 2*np.pi,

        'source_isocenter_distance': 750,
        'source_detector_distance': 1200
    }

    geometry = MySpecificGeometry(geo_info)

    # mask, sinogram, phantom, geo = geometry.generate_specific_phantom(primitives_3d.generate_3D_primitives, number_of_primitives=6)
    mask, sinogram, phantom, geo = geometry.generate_specific_phantom(shepp_logan.shepp_logan_3d)
    # ------------------ Call Layers ------------------
    # The following code is the new TF2.0 experimental way to tell
    # Tensorflow only to allocate GPU memory needed rather then allocate every GPU memory available.
    # This is important for the use of the hardware interpolation projector, otherwise there might be not enough memory left
    # to allocate the texture memory on the GPU

    reco_filter = shepp_logan_3D(geo.detector_shape, geo.detector_spacing, geo.number_of_projections)
    x = fft_and_ifft(sinogram, reco_filter)

    reco = ConeBackProjectionFor3D().forward(x, geo)

    plt.figure()
    plt.imshow(np.squeeze(reco)[geo.volume_shape[0] // 2], cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig(f'3d_cone_reco_diy_geo.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_specific_geometry()
