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
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.trajectories          import circular_trajectory


"""
    This file defines the Geometry parameters used by the hole model. 
    A GeometryParallel2D instance is provided to be used by everyone that needs it.
"""

# Declare Parameters
volume_shape          = [256, 256]
volume_spacing        = [0.5, 0.5]
detector_shape        = 365
detector_spacing      = 0.5
number_of_projections = 720
angular_range         = 2*np.pi

# Create Geometry class instance
GEOMETRY = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
GEOMETRY.set_ray_vectors(circular_trajectory.circular_trajectory_2d(GEOMETRY))
