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

import pyronn_layers
import numpy as np

from torch import Tensor
from torch import nn
from torch.autograd import Function
import torch

import pyronn_layers

class ParallelProjection2DFunction(Function):
    @staticmethod
    def forward(ctx, input:Tensor, sinogram_shape:Tensor, volume_origin:Tensor, detector_origin:Tensor, volume_spacing:Tensor, detector_spacing:Tensor, trajectory)->Tensor:
        """
        Forward operator of 2D parallel projection
        Args: 
                input:              volume to be projected
                sinogram_shape:     number_of_projections x detector_width
                volume_origin:      origin of the world coordinate system w.r.t. the volume array (tensor)
                ...
        """
        outputs = pyronn_layers.parallel_projection2d(input,sinogram_shape, volume_origin,detector_origin,volume_spacing,detector_spacing,trajectory)
        
        ctx.volume_shape        = torch.tensor(input.shape[1:]).cuda()
        ctx.volume_origin       = volume_origin
        ctx.detector_origin     = detector_origin
        ctx.volume_spacing      = volume_spacing
        ctx.detector_spacing    = detector_spacing
        ctx.trajectory          = trajectory

        return outputs

    @staticmethod
    def backward(ctx, grad:Tensor)->tuple:
        # volume_shape, volume_origin, detector_origin, volume_spacing, detector_spacing, trajectory = ctx.saved_tensors
        volume_shape        = ctx.volume_shape
        volume_origin       = ctx.volume_origin
        detector_origin     = ctx.detector_origin
        volume_spacing      = ctx.volume_spacing
        detector_spacing    = ctx.detector_spacing
        trajectory          = ctx.trajectory
        if not grad.is_contiguous():
            grad = grad.contiguous()
        outputs = pyronn_layers.parallel_backprojection2d(grad,
                                                                volume_shape,
                                                                volume_origin,
                                                                detector_origin,
                                                                volume_spacing,
                                                                detector_spacing,                                                                
                                                                trajectory)
        d_input = outputs
        
        return d_input, None, None, None, None, None, None


class ParallelProjection2D(nn.Module):
    def __init__(self):
        super(ParallelProjection2D, self).__init__()

    def forward(self, input:Tensor, **geometry:dict)->Tensor:
        return ParallelProjection2DFunction.apply(input, geometry['sinogram_shape'],geometry['volume_origin'],geometry['detector_origin'],geometry['volume_spacing'],geometry['detector_spacing'],geometry['trajectory'])

class FanProjection2DFunction(Function):
    @staticmethod
    def forward(ctx, input:Tensor, sinogram_shape:Tensor, volume_origin:Tensor, detector_origin:Tensor, volume_spacing:Tensor, detector_spacing:Tensor, source_isocenter_distance:Tensor, source_detector_distance:Tensor, trajectory:Tensor)->Tensor:
        """
        Forward operator of 2D fan projection
        Args: 
                input:              volume to be projected
                sinogram_shape:     number_of_projections x detector_width
                volume_origin:      origin of the world coordinate system w.r.t. the volume array (tensor)
                ...
        """
        outputs = pyronn_layers.fan_projection2d(input,sinogram_shape, volume_origin,detector_origin,volume_spacing,detector_spacing,source_isocenter_distance,source_detector_distance,trajectory)
    
        ctx.volume_shape        = torch.tensor(input.shape[1:]).cuda()
        ctx.volume_origin       = volume_origin
        ctx.detector_origin     = detector_origin
        ctx.volume_spacing      = volume_spacing
        ctx.detector_spacing    = detector_spacing
        ctx.source_isocenter_distance      = source_isocenter_distance
        ctx.source_detector_distance    = source_detector_distance
        ctx.trajectory          = trajectory

        return outputs

    @staticmethod
    def backward(ctx, grad:Tensor)->tuple:
        volume_shape                = ctx.volume_shape
        volume_origin               = ctx.volume_origin
        detector_origin             = ctx.detector_origin
        volume_spacing              = ctx.volume_spacing
        detector_spacing            = ctx.detector_spacing
        source_isocenter_distance   = ctx.source_isocenter_distance
        source_detector_distance    = ctx.source_detector_distance
        trajectory                  = ctx.trajectory
        if not grad.is_contiguous():
            grad = grad.contiguous()
        outputs = pyronn_layers.fan_backprojection2d(grad,
                                                                volume_shape,
                                                                volume_origin,
                                                                detector_origin,
                                                                volume_spacing,
                                                                detector_spacing,   
                                                                source_isocenter_distance,
                                                                source_detector_distance,                                                             
                                                                trajectory)
        d_input = outputs
        
        return d_input, None, None, None, None, None, None, None, None


class FanProjection2D(nn.Module):
    def __init__(self):
        super(FanProjection2D, self).__init__()

    def forward(self, input:Tensor, **geometry:dict)->Tensor:
        return FanProjection2DFunction.apply(input, geometry['sinogram_shape'], geometry['volume_origin'], geometry['detector_origin'], geometry['volume_spacing'], geometry['detector_spacing'],
                                                    geometry['source_isocenter_distance'], geometry['source_detector_distance'], geometry['trajectory'])
    