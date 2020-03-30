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


def circular_trajectory_2d(geometry):
    """
        Generates the central ray vectors defining a circular trajectory for use with the 2d projection layers.
    Args:
        geometry: 2d Geometry class including angular_range and number_of_projections
    Returns:
        Central ray vectors as np.array.
    """
    rays = np.zeros([geometry.number_of_projections, 2])
    angular_increment = geometry.angular_range / geometry.number_of_projections
    for i in range(geometry.number_of_projections):
        rays[i] = [np.cos(i * angular_increment), np.sin(i * angular_increment)]
    return rays


def circular_trajectory_3d(geometry):
    """
        Generates the projection matrices defining a circular trajectory around the z-axis
        for use with the 3d projection layers.
        Adapted from CONRAD Source code https://github.com/akmaier/CONRAD.
    Args:
        geometry: 3d Geometry class including angular_range, number_of_projections, source_detector_distance,
        detector_shape, detector_spacing, volume_origin, volume_shape and volume_spacing.
    Returns:
        Projection matrices with shape (num_projections, 3, 4) as np.array.
    """

    # init empty
    projection_matrices = np.zeros((geometry.number_of_projections, 3, 4))

    # axes for later use
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    # defining u and v directions by: main coord axes
    u_dir = y_axis
    v_dir = -x_axis

    # configure intrinsic camera parameters
    intrinsic_params_mat = np.eye(3, 3)
    for i in range(2):
        intrinsic_params_mat[i, i] = geometry.source_detector_distance / geometry.detector_spacing[1-i]

    # calc and set detector origin
    intrinsic_params_mat[0:2, 2] = (geometry.detector_shape * 0.5)[::-1]

    # configure extrinisc pararams and create projection_matrices
    current_angle = 0.0
    angular_increment = geometry.angular_range / geometry.number_of_projections
    for p in range(geometry.number_of_projections):
        # calculate extrinsic params
        extrinsic_params_mat = np.eye(4, 4)

        # rotation of axes from world system to plane of rotation system
        R_to_plane = np.eye(4, 4)
        R_to_plane[0:3, 0:3] = np.array([z_axis, np.cross(z_axis, x_axis), -x_axis])

        # rotation for u and v direction
        axis_align_R = np.eye(4, 4)
        axis_align_R[0:3, 0] = u_dir
        axis_align_R[0:3, 1] = v_dir
        axis_align_R[0:3, 2] = np.cross(u_dir, v_dir)
        axis_align_R = axis_align_R.T

        # rotation about x axis
        R_x_axis = np.eye(4, 4)
        R_x_axis[0:3, 0:3] = np.array([1, 0, 0,
                                       0, np.cos(-current_angle), -np.sin(-current_angle),
                                       0, np.sin(-current_angle), np.cos(-current_angle)]).reshape((3, 3))

        # translation of camera
        translation = np.eye(4, 4)
        translation[0:4, 3] = np.array([0, 0, geometry.source_isocenter_distance, 1])

        # combine the above into 4x4 extrinsic params matrix
        extrinsic_params_mat = np.dot(np.dot(np.dot(translation, axis_align_R), R_x_axis), R_to_plane)
        extrinsic_params_mat = extrinsic_params_mat / extrinsic_params_mat[3, 3]

        # calculate projection matrix
        projection_matrices[p][0:3, 0:3] = np.dot(intrinsic_params_mat, extrinsic_params_mat[0:3, 0:3])
        projection_matrices[p][0:3, 3] = np.dot(intrinsic_params_mat, extrinsic_params_mat[0:3, 3])

        # next angle
        current_angle += angular_increment

    return projection_matrices
