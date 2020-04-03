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

import tensorflow as tf
from ...layers.projection_2d import parallel_projection2d
from ...layers.projection_2d import fan_projection2d
from ...layers.projection_3d import cone_projection3d


def generate_sinogram(phantom, layer, geometry):

    result = layer(phantom, geometry)
    return result


def generate_sinogram_parallel_2d(phantom, geometry):
    result = parallel_projection2d(phantom, geometry)
    return result


def generate_sinogram_fan_2d(phantom, geometry):
    result = fan_projection2d(phantom, geometry)
    return result


def generate_sinogram_cone_3d(phantom, geometry):
    result = cone_projection3d(phantom, geometry)
    return result
