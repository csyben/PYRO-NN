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

from tensorflow.python.framework import ops
import pyronn_layers


# cone_projection3d
def cone_projection3d(volume, geometry, hardware_interp=True, step_size=1.0):
    """
    Wrapper function for making the layer call.
    Args:
        volume:             Input volume to project.
        geometry:           Corresponding GeometryCone3D Object defining parameters.
        hardware_interp:    Controls if interpolation is done by GPU.
        step_size:          step_size along ray direction in voxel.
    Returns:
            Initialized lme_custom_ops.cone_projection3d layer.
    """

    return pyronn_layers.cone_projection3d(volume,
                                            volume_shape=geometry.volume_shape,
                                            projection_shape=geometry.sinogram_shape,
                                            volume_origin=geometry.tensor_proto_volume_origin,
                                            volume_spacing=geometry.tensor_proto_volume_spacing,
                                            projection_matrices=geometry.tensor_proto_projection_matrices,
                                            hardware_interp=hardware_interp,
                                            step_size=step_size,
                                            projection_multiplier = geometry.projection_multiplier)


@ops.RegisterGradient("ConeProjection3D")
def _project_grad(op, grad):
    '''
    Compute the gradient of the projection op by invoking the backprojector.
    '''
    reco = pyronn_layers.cone_backprojection3d(
        sinogram=grad,
        sinogram_shape=op.get_attr("projection_shape"),
        volume_shape=op.get_attr("volume_shape"),
        volume_origin=op.get_attr("volume_origin"),
        volume_spacing=op.get_attr("volume_spacing"),
        projection_multiplier=op.get_attr("projection_multiplier"),
        projection_matrices=op.get_attr("projection_matrices"),
        hardware_interp=op.get_attr("hardware_interp"),
        step_size=op.get_attr("step_size")
    )
    return [reco]
