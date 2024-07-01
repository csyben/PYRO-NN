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
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pyronn.ct_reconstruction.layers.torch.projection_3d import ConeProjection3D
from pyronn.ct_reconstruction.layers.torch.backprojection_3d import ConeBackProjection3D
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak_3D
import pyronn.ct_reconstruction.helpers.filters.weights as ct_weights
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d


class nn_model:
    def __init__(self, geometry):

        filter = ram_lak_3D(detector_shape=geometry['detector_shape'],
                            detector_spacing=geometry['detector_spacing'],
                            number_of_projections=geometry['number_of_projections'])
        self.geometry = geometry

        cosine_weight = ct_weights.cosine_weights_3d(geometry)
        self.cosine_weight = nn.Parameter(torch.tensor(np.copy(cosine_weight), dtype=torch.float32), requires_grad=False)

        parker_weight = ct_weights.parker_weights_3d(geometry)
        self.redundancy_weight = nn.Parameter(torch.tensor(np.copy(parker_weight), dtype=torch.float32), requires_grad=False)


        self.filter = nn.Parameter(torch.tensor(filter, dtype=torch.float32), requires_grad=False)



    def model(self, sinogram):
        self.sinogram_cos = torch.multiply(sinogram, self.cosine_weight)
        self.redundancy_weighted_sino = torch.multiply(self.sinogram_cos,self.redundancy_weight)

        self.weighted_sino_fft = torch.fft.fft(self.redundancy_weighted_sino.to(torch.complex64))
        self.filtered_sinogram_fft = torch.multiply(self.weighted_sino_fft, self.filter.to(torch.complex64))
        self.filtered_sinogram = torch.real(torch.fft.ifft(self.filtered_sinogram_fft))

        self.reconstruction = ConeBackProjection3D.forward(self.filtered_sinogram,self.geometry)

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
    geometry = Geometry()
    geometry.init_from_parameters(volume_shape=volume_shape, volume_spacing=volume_spacing,
                                  detector_shape=detector_shape, detector_spacing=detector_spacing,
                                  number_of_projections=number_of_projections, angular_range=angular_range,
                                  trajectory=circular_trajectory_3d, source_isocenter_distance=source_isocenter_distance,
                                  source_detector_distance=source_detector_distance)

    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    phantom = torch.tensor(np.expand_dims(phantom, axis=0).copy(), dtype=torch.float32).cuda()

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RunetimeError as e:
    #         print(e)
    # ------------------ Call Layers ------------------
    sinogram = ConeProjection3D().forward(phantom, **geometry)

    model = nn_model(geometry)
    reco, redundancy_weighted_sino = model.model(sinogram)

    plt.figure()
    plt.imshow(np.squeeze(reco)[volume_shape[0]//2], cmap=plt.get_cmap('gist_gray'), vmin=0, vmax=0.4)
    plt.axis('off')
    plt.savefig('fdk_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_cone_3d()
