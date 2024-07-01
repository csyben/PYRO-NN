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

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Function
import pyronn_layers
import numpy as np

# cone_projection3d
class ConeProjection3DFunction(Function):
    @staticmethod
    def forward(ctx, input:Tensor, sinogram_shape:Tensor, volume_origin:Tensor, volume_spacing:Tensor, trajectory:Tensor,
                     projection_multiplier:Tensor, step_size:Tensor, hardware_interp:Tensor)->Tensor:
        """
        Forward operator of 2D fan projection
        Args: 
                input:              volume to be projected
                sinogram_shape:     number_of_projections x detector_width
                volume_origin:      origin of the world coordinate system w.r.t. the volume array (tensor)
                ...
        """
        outputs = pyronn_layers.cone_projection3d(input,sinogram_shape, volume_origin,volume_spacing,trajectory, step_size, hardware_interp)
        
        ctx.volume_shape            = torch.tensor(input.shape[1:]).cuda()
        ctx.volume_origin           = volume_origin
        ctx.volume_spacing          = volume_spacing
        ctx.trajectory              = trajectory
        ctx.projection_multiplier   = projection_multiplier
        ctx.hardware_interp         = hardware_interp

        return outputs

    @staticmethod
    def backward(ctx, grad:Tensor)->tuple:
        volume_shape            = ctx.volume_shape
        volume_origin           = ctx.volume_origin
        volume_spacing          = ctx.volume_spacing
        trajectory              = ctx.trajectory
        projection_multiplier   = ctx.projection_multiplier
        hardware_interp         = ctx.hardware_interp
        if not grad.is_contiguous():
            grad = grad.contiguous()
        outputs = pyronn_layers.cone_backprojection3d(grad,
                                                                volume_shape,
                                                                volume_origin,                                                                
                                                                volume_spacing,                                                             
                                                                trajectory,
                                                                projection_multiplier,
                                                                hardware_interp)
        d_input = outputs
        
        return d_input, None, None, None, None, None, None, None


class ConeProjection3D(nn.Module):
    def __init__(self, hardware_interp = False):
        super(ConeProjection3D, self).__init__()
        self.hardware_interp = torch.Tensor([hardware_interp]).cpu()

    def forward(self, input:Tensor, **geometry:dict)->Tensor:
        return ConeProjection3DFunction.apply(input, geometry['sinogram_shape'], geometry['volume_origin'], geometry['volume_spacing'], geometry['trajectory'], geometry['projection_multiplier'], geometry['step_size'], self.hardware_interp)