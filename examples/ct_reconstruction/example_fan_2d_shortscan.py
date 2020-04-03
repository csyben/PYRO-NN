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
from pyronn.ct_reconstruction.geometry.geometry_fan_2d                 import GeometryFan2D
from pyronn.ct_reconstruction.helpers.filters                          import filters, weights
from pyronn.ct_reconstruction.helpers.phantoms.shepp_logan             import shepp_logan_enhanced
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d
from pyronn.ct_reconstruction.layers.backprojection_2d                 import fan_backprojection2d
from pyronn.ct_reconstruction.layers.projection_2d                     import fan_projection2d


def example_fan_2d_shortscan():
    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1,1]

    # Detector Parameters:
    detector_shape = 500
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = 250
    angular_range = None # will get set to pi + 2 * fan_angle

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # Create Geometry class
    geometry = GeometryFan2D(volume_shape, volume_spacing, 
                             detector_shape, detector_spacing, 
                             number_of_projections, angular_range, 
                             source_detector_distance, source_isocenter_distance)
    
    geometry.angular_range =  np.pi + 2*geometry.fan_angle # fan_angle gets defined by sdd and detector_shape
    geometry.set_trajectory(circular_trajectory_2d(geometry))

    # Create Phantom
    phantom = shepp_logan_enhanced(volume_shape)
    # Add required batch dimension
    phantom = np.expand_dims(phantom,axis=0)
    # Build up Reconstruction Pipeline

    # Create Sinogram of Phantom
    sinogram = fan_projection2d(phantom, geometry)

    # Redundancy Weighting: Create Weights Image and pointwise multiply
    redundancy_weights = weights.parker_weights_2d(geometry)
    sinogram_redun_weighted = sinogram * redundancy_weights

    # Filtering: Create 2D Filter and pointwise multiply
    reco_filter = filters.ram_lak_2D(geometry)
    sino_freq = tf.signal.fft(tf.cast(sinogram_redun_weighted,dtype=tf.complex64))
    sino_filtered_freq = tf.multiply(sino_freq,tf.cast(reco_filter,dtype=tf.complex64))
    sinogram_filtered = tf.math.real(tf.signal.ifft(sino_filtered_freq))
    # Final Backprojection
    reco = fan_backprojection2d(sinogram_filtered, geometry)

    plt.figure()
    plt.imshow(np.squeeze(reco), cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig('2d_fan_short_scan_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_fan_2d_shortscan()
