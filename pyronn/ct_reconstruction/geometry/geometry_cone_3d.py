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
from .geometry_base import GeometryBase


class GeometryCone3D(GeometryBase):
    """
        3D Cone specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range,
                 source_detector_distance, source_isocenter_distance):
        # init base Geometry class with 3 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         detector_shape, detector_spacing,
                         number_of_projections, angular_range,
                         source_detector_distance, source_isocenter_distance)

        # defined by geometry so calculate for convenience use
        self.fan_angle  = np.arctan(((self.detector_shape[1] - 1) / 2.0 * self.detector_spacing[1]) / self.source_detector_distance)
        self.cone_angle = np.arctan(((self.detector_shape[0] - 1) / 2.0 * self.detector_spacing[0]) / self.source_detector_distance)

        # Containing the constant part of the distance weight and discretization invariant
        self.projection_multiplier = self.source_isocenter_distance * self.source_detector_distance * detector_spacing[-1] * np.pi / self.number_of_projections
        self.step_size = 1.0

    def set_trajectory(self, projection_matrices):
        """
            Sets the member projection_matrices.
        Args:
            projection_matrices: np.array defining the trajectory projection_matrices.
        """
        self.projection_matrices = np.array(projection_matrices, self.np_dtype)

