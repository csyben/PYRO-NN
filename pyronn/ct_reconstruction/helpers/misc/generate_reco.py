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
from ...layers.backprojection_2d import parallel_backprojection2d
from ...layers.backprojection_2d import fan_backprojection2d
from ...layers.backprojection_3d import cone_backprojection3d


def generate_reco(sinogram, layer, geometry):
    with tf.Session() as sess:
        result = layer(sinogram, geometry)
        sinogram = result.eval()
    return sinogram


def generate_reco_parallel_2d(phantom, geometry):
    result = parallel_backprojection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram


def generate_reco_fan_2d(phantom, geometry):
    result = fan_backprojection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram


def generate_reco_cone_3d(phantom, geometry):
    result = cone_backprojection3d(phantom, geometry)
    sinogram = result.eval()
    return sinogram
