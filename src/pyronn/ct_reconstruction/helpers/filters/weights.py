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


# 3d cosine weights
def cosine_weights_3d(geometry):
    cu = -(geometry.detector_shape[-1] - 1) / 2 * geometry.detector_spacing[-1]
    cv = -(geometry.detector_shape[-2] - 1) / 2 * geometry.detector_spacing[-2]
    sd2 = geometry.source_detector_distance ** 2

    w = np.zeros((geometry.detector_shape[-2], geometry.detector_shape[-1]), dtype=np.float32)

    for v in range(0, geometry.detector_shape[-2]):
        dv = (v * geometry.detector_spacing[-2] + cv) ** 2
        for u in range(0, geometry.detector_shape[-1]):
            du = (u * geometry.detector_spacing[-1] + cu) ** 2
            w[v, u] = geometry.source_detector_distance / np.sqrt(sd2 + dv + du)

    return np.flip(w)


def parker_weights_3d(geometry):
    weights = np.flip(parker_weights_2d(geometry), axis=1)
    weights = np.array(np.expand_dims(weights, axis=1), dtype=np.float32)
    return weights


def parker_weights_2d(geometry):
    number_of_projections = geometry.number_of_projections
    angular_range = geometry.angular_range[1] - geometry.angular_range[0]
    detector_shape = geometry.detector_shape
    detector_spacing = geometry.detector_spacing
    # detector_origin = geometry.detector_origin
    source_detector_distance = geometry.source_detector_distance
    fan_angle = geometry.fan_angle

    weights = np.ones((number_of_projections, detector_shape[-1]))
    angular_increment = angular_range / number_of_projections
    beta = 0
    beta = ((np.pi + 2*fan_angle) - angular_range) / 2.0 # adds offset

    for beta_idx in range(weights.shape[0]):
        for gamma_idx in range(weights.shape[1]):
                # calculate correct pos on detector and current angle
                gamma_angle = gamma_idx * detector_spacing[-1]# + detector_origin[-1]
                gamma_angle = np.arctan(gamma_angle / source_detector_distance)

                # check if rays sampled twice and create weight volume
                if 0 <= beta and beta <= 2*(fan_angle - gamma_angle):
                    val = np.sin( ((np.pi/4.0) * beta) / (fan_angle - gamma_angle) ) ** 2
                    if not np.isnan(val):
                        weights[beta_idx, gamma_idx] = val

                elif 2*(fan_angle - gamma_angle) < beta and beta < np.pi - 2*gamma_angle:
                    weights[beta_idx, gamma_idx] = 1.0

                elif np.pi - 2*gamma_angle <= beta and beta <= np.pi + 2*fan_angle:
                    val = np.sin((np.pi/4.0) * ((np.pi + 2*fan_angle - beta) / (gamma_angle + fan_angle))) ** 2
                    if not np.isnan(val):
                        weights[beta_idx, gamma_idx] = val

                else:
                    weights[beta_idx, gamma_idx] = 0
        
        beta += angular_increment

    # additional scaling factor
    scale_factor = (angular_range + angular_range  / number_of_projections) / np.pi

    return weights * scale_factor


# Adapted from:
# TV or not TV? That is the Question
# Christian Riess, Martin Berger, Haibo Wu, Michael Manhart, Rebecca Fahrig and Andreas Maier
# The 12th International Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine
# Note: This is the unsmoothed version of the by Riess, et al proposed weights, which may lead to artefacts.
def riess_weights_2d(geometry):

    delta_x = geometry.angular_range - np.pi # overscan

    def eta(beta, gamma_angle):
        return np.sin( (np.pi/2.0) * (np.pi+delta_x-beta) / (delta_x-2*gamma_angle) ) ** 2

    def zeta(beta, gamma_angle):
        return np.sin( (np.pi/2.0) * beta / (delta_x+2*gamma_angle) ) ** 2

    weights = np.ones((geometry.number_of_projections, geometry.detector_shape[-1]))
    angular_increment = geometry.angular_range / geometry.number_of_projections
    beta = 0

    for beta_idx in range(weights.shape[0]):
        for gamma_idx in range(weights.shape[1]):
                # calculate correct pos on detector and current angle
                gamma_angle = gamma_idx * geometry.detector_spacing[-1] + geometry.detector_origin[-1]
                gamma_angle = np.arctan(gamma_angle / geometry.source_detector_distance)

                if np.pi + 2*gamma_angle <= beta and beta <= np.pi + delta_x:
                    val = eta(beta, gamma_angle)
                    if not np.isnan(val):
                        weights[beta_idx, gamma_idx] = val

                if np.pi + 2*(delta_x - gamma_angle) <= beta and beta <= np.pi + delta_x:
                    val = 2 - eta(beta, gamma_angle)
                    if not np.isnan(val):
                        weights[beta_idx, gamma_idx] = val

                if 0 <= beta and beta <= 2*gamma_angle + delta_x:
                    val = zeta(beta, gamma_angle)
                    if not np.isnan(val):
                        weights[beta_idx, gamma_idx] = val

                if 0 <= beta and beta <= -delta_x - 2*gamma_angle:
                    val = 2 - zeta(beta, gamma_angle)
                    if not np.isnan(val):
                        weights[beta_idx, gamma_idx] = val

        beta += angular_increment

    # additional scaling factor
    scale_factor = geometry.angular_range / np.pi
    return weights * scale_factor
