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
import numpy as np
import tensorflow as tf

# cone_backprojection3d
def cone_backprojection3d(sinogram, geometry, hardware_interp=True):
    """
    Wrapper function for making the layer call.
    Args:
        volume:             Input volume to project.
        geometry:           Corresponding GeometryCone3D Object defining parameters.
        hardware_interp:    Controls if interpolation is done by GPU 
    Returns:
            Initialized lme_custom_ops.cone_backprojection3d layer.
    """
    batch = np.shape(sinogram)[0]
    step_size=1.0
    return pyronn_layers.cone_backprojection3d(sinogram,
                                               volume_shape=geometry.volume_shape,
                                               volume_origin=np.broadcast_to(geometry.volume_origin, [batch, *np.shape(geometry.volume_origin)]),
                                               volume_spacing=np.broadcast_to(geometry.volume_spacing, [batch, *np.shape(geometry.volume_spacing)]),
                                               projection_matrices=np.broadcast_to(geometry.trajectory, [batch, *np.shape(geometry.trajectory)]),
                                               hardware_interp=hardware_interp,
                                               step_size=np.broadcast_to(step_size, [batch, *np.shape(step_size)]),
                                               projection_multiplier=np.broadcast_to(geometry.projection_multiplier, [batch, *np.shape(geometry.projection_multiplier)]))


@ops.RegisterGradient("ConeBackprojection3D")
def _backproject_grad(op, grad):
    '''
        Compute the gradient of the backprojector op by invoking the forward projector.
    '''
    proj = pyronn_layers.cone_projection3d(
        volume=grad,
        projection_shape=op.inputs[0].shape[1:],
        volume_origin=op.inputs[2],
        volume_spacing=op.inputs[3],
        projection_matrices=op.inputs[4],
        hardware_interp=op.get_attr("hardware_interp"),
        step_size=op.inputs[5],
        projection_multiplier=op.inputs[6],
    )
    return [proj, tf.stop_gradient(op.inputs[1]), tf.stop_gradient(op.inputs[2]), tf.stop_gradient(op.inputs[3]), tf.stop_gradient(op.inputs[4]), tf.stop_gradient(op.inputs[5]), tf.stop_gradient(op.inputs[6])]

