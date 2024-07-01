import numpy as np

import torch
from torch import nn
from pyronn.ct_reconstruction.geometry.geometry import Geometry


class ParallelBeamReconstruction2D(nn.Module):
    def __init__(self, geometry:Geometry, filter:np.ndarray, tainable_filter:bool=False)->None:
        """
            - short scan yes no?
            - trainable reko filter
            - filter type ?
            - trainable redundancy weights
            -

        """
        from pyronn.ct_reconstruction.layers.torch.backprojection_2d import ParallelBackProjection2D
        super(ParallelBeamReconstruction2D, self).__init__()
        self.filter = nn.Parameter(torch.tensor(filter, requires_grad=tainable_filter,dtype=torch.cfloat).cuda())
        self.backprojection = ParallelBackProjection2D()
    def forward(self, input, **kwargs):
        x = torch.fft.fft(input,dim=-1,norm="ortho")
        x = torch.multiply(x,self.filter)
        x = torch.fft.ifft(x,dim=-1,norm="ortho")
        x = x.real
        if not x.is_contiguous():
            x = x.contiguous()
        x = self.backprojection.forward(x,**kwargs)
        return x


class FanBeamReconstruction(nn.Module):
    def __init__(self, geometry:Geometry, filter:np.ndarray, short_scan=False, redundancy_weights:np.ndarray=None, trainable_filter=False, trainable_redundancy_weights=False):
        """
            - short scan yes no?
            - trainable reko filter
            - filter type ?
            - trainable redundancy weights
            -

        """
        from pyronn.ct_reconstruction.layers.torch.backprojection_2d import FanBackProjection2D
        super(FanBeamReconstruction, self).__init__()
        self.filter = nn.Parameter(torch.tensor(filter, requires_grad=trainable_filter).cuda())
        if short_scan:
            self.redundancy_weights = nn.Parameter(torch.tensor(redundancy_weights, requires_grad=trainable_redundancy_weights).cuda())
        self.backprojection = FanBackProjection2D()
    def forward(self, input, **kwargs):
        x = input
        if self.redundancy_weights is not None:
            x = torch.multiply(x,self.redundancy_weights)
        x = torch.fft.fft(input,dim=-1,norm="ortho")
        x = torch.multiply(x,self.filter)
        x = torch.fft.ifft(x,dim=-1,norm="ortho")
        x = x.real
        if not x.is_contiguous():
            x = x.contiguous()
        x = self.backprojection.forward(x,**kwargs)
        return x

class FDKConeBeamReconstruction(nn.Module):
    def __init__(self, geometry:Geometry, filter:np.ndarray, short_scan=False, trainable_redundancy_weights:np.ndarray=None, trainable_filter = False):
        a=5