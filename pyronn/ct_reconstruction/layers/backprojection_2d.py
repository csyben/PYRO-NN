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


# parallel_backprojection2d
def parallel_backprojection2d(sinogram, geometry):
    """
    Wrapper function for making the layer call.
    Args:
        volume:     Input volume to project.
        geometry:   Corresponding GeometryParallel2D Object defining parameters.
    Returns:
            Initialized lme_custom_ops.parallel_backprojection2d layer.
    """
    batch = np.shape(sinogram)[0]
    return pyronn_layers.parallel_backprojection2d(sinogram,
                                                   volume_shape=geometry.volume_shape,
                                                    volume_origin   =np.broadcast_to(geometry.volume_origin,[batch,*np.shape(geometry.volume_origin)]),
                                                    detector_origin =np.broadcast_to(geometry.detector_origin,[batch,*np.shape(geometry.detector_origin)]),
                                                    volume_spacing  =np.broadcast_to(geometry.volume_spacing,[batch,*np.shape(geometry.volume_spacing)]),
                                                    detector_spacing=np.broadcast_to(geometry.detector_spacing,[batch,*np.shape(geometry.detector_spacing)]),
                                                    ray_vectors     =np.broadcast_to(geometry.ray_vectors,[batch,*np.shape(geometry.ray_vectors)]))


@ops.RegisterGradient("ParallelBackprojection2D")
def _backproject_grad(op, grad):
    '''
        Compute the gradient of the backprojector op by invoking the forward projector.
    '''
    proj = pyronn_layers.parallel_projection2d(
        volume=grad,
        projection_shape=op.inputs[0].shape[1:],
        volume_origin=op.inputs[2],
        detector_origin=op.inputs[3],
        volume_spacing=op.inputs[4],
        detector_spacing=op.inputs[5],
        ray_vectors=op.inputs[6],
    )
    return [proj, tf.stop_gradient(op.inputs[1]), tf.stop_gradient(op.inputs[2]), tf.stop_gradient(op.inputs[3]), tf.stop_gradient(op.inputs[4]), tf.stop_gradient(op.inputs[5]), tf.stop_gradient(op.inputs[6])]


# fan_backprojection2d
def fan_backprojection2d(sinogram, geometry):
    """
    Wrapper function for making the layer call.
    Args:
        volume:     Input volume to project.
        geometry:   Corresponding GeometryFan2D Object defining parameters.
    Returns:
            Initialized lme_custom_ops.fan_backprojection2d layer.
    """
    batch = np.shape(sinogram)[0]
    return pyronn_layers.fan_backprojection2d(sinogram,
                                              volume_shape=geometry.volume_shape,
                                              volume_origin=np.broadcast_to(geometry.volume_origin, [batch, *np.shape(geometry.volume_origin)]),
                                              detector_origin=np.broadcast_to(geometry.detector_origin, [batch, *np.shape(geometry.detector_origin)]),
                                              volume_spacing=np.broadcast_to(geometry.volume_spacing, [batch, *np.shape(geometry.volume_spacing)]),
                                              detector_spacing=np.broadcast_to(geometry.detector_spacing, [batch, *np.shape(geometry.detector_spacing)]),
                                              source_2_isocenter_distance=np.broadcast_to(geometry.source_isocenter_distance, [batch, *np.shape(geometry.source_isocenter_distance)]),
                                              source_2_detector_distance=np.broadcast_to(geometry.source_detector_distance, [batch, *np.shape(geometry.source_detector_distance)]),
                                              central_ray_vectors=np.broadcast_to(geometry.central_ray_vectors, [batch, *np.shape(geometry.central_ray_vectors)]))


@ops.RegisterGradient("FanBackprojection2D")
def _backproject_grad(op, grad):
    '''
        Compute the gradient of the backprojector op by invoking the forward projector.
    '''
    proj = pyronn_layers.fan_projection2d(
        volume=grad,
        projection_shape=op.inputs[0].shape[1:],
        volume_origin=op.inputs[2],
        detector_origin=op.inputs[3],
        volume_spacing=op.inputs[4],
        detector_spacing=op.inputs[5],
        source_2_isocenter_distance=op.inputs[6],
        source_2_detector_distance=op.inputs[7],
        central_ray_vectors=op.inputs[8],
    )
    return [proj, tf.stop_gradient(op.inputs[1]),  tf.stop_gradient(op.inputs[2]),  tf.stop_gradient(op.inputs[3]), tf.stop_gradient(op.inputs[4]),
            tf.stop_gradient(op.inputs[5]),  tf.stop_gradient(op.inputs[6]),  tf.stop_gradient(op.inputs[7]), tf.stop_gradient(op.inputs[8])]
