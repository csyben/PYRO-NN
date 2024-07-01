import torch
import numpy as np
from torch import nn
import kornia
from pyronn.ct_reconstruction.helpers.filters.filters import ramp, shepp_logan_3D

from pyronn.ct_reconstruction.layers.torch.backprojection_3d import ConeBackProjection3D

from pyronn.ct_reconstruction.geometry.geometry import Geometry
from PythonTools import raw2py

class Reconstruction(nn.Module):
    def __init__(self, shape):
        super(Reconstruction, self).__init__()
        self.filter = nn.Parameter(torch.tensor(ramp(shape[-1]),dtype=torch.float32))
        # self.filter = nn.Parameter(torch.tensor(ramp_3D(geometry),dtype=torch.float32))
        # self.filter = nn.Parameter(torch.zeros(1,1,shape[-1], requires_grad=True).cuda())#Filter parameter with batch (1,1,detector_width)
        # self.backprojection = ParallelBackProjection2D()
        # self.backprojection = FanBackProjection2D()
        self.backprojection = ConeBackProjection3D()
    def forward(self, input, **kwargs):
        x = torch.fft.fft(input,dim=-1,norm="ortho")
        x = torch.multiply(x,self.filter)
        x = torch.fft.ifft(x,dim=-1,norm="ortho")
        x = x[:,0:500,:,:]
        x = self.backprojection.forward(x.real.cuda().contiguous(),**kwargs)
        return x




# def circular_trajectory_2d_test(number_of_projections, angular_range):
#     rays = np.zeros([number_of_projections, 2])
#     angular_increment = angular_range / number_of_projections
#     for i in range(number_of_projections):
#         rays[i] = [np.cos(i * angular_increment), np.sin(i * angular_increment)]
#     return rays

# def get_geometry(volume_shape, volume_origin, volume_spacing, sinogram_shape, detector_origin, detector_spacing, trajectory):
#     geometry = {}
#     geometry.setdefault('volume_shape',volume_shape)
#     geometry.setdefault('sinogram_shape',sinogram_shape)
#     geometry.setdefault('volume_origin',volume_origin)
#     geometry.setdefault('detector_origin',detector_origin)
#     geometry.setdefault('detector_origin',detector_origin)
#     geometry.setdefault('volume_spacing',volume_spacing)
#     geometry.setdefault('detector_spacing',detector_spacing)
#     geometry.setdefault('trajectory',trajectory)
#     return geometry

# def in_call(input, sinogram_shape, volume_origin,detector_origin,volume_spacing,detector_spacing,trajectory):
#     print(sinogram_shape.shape)
#     print(volume_origin.shape)
#     print(detector_origin.shape)
#     print(volume_spacing.shape)
#     print(detector_spacing.shape)
#     print(trajectory.shape)
#     a=5


# def in_test(volume,**geometry):
#     in_call(input, geometry['sinogram_shape'],geometry['volume_origin'],geometry['detector_origin'],geometry['volume_spacing'],geometry['detector_spacing'],geometry['trajectory'])

#Start
# shape = [256,256]
# shape= [256,256,256]
# image = shepp_logan_enhanced(np.asarray(shape))
# image1 = rect(np.asarray(shape),[64,64],[100,100])
# image2 = circle(np.asarray(shape),[128,128],100)
# image1 = cube(np.asarray(shape),[64,64,64],[100,100,100])
# image2 = sphere(np.asarray(shape),[128,128,128],100)
#
#
# number_of_projections = 180
# detector_width = 365
# detector_height = 365
# detector_shape = [detector_width]
# detector_shape = [detector_height,detector_width]
# angular_range = np.pi
#
# volume = torch.tensor([image], dtype=torch.float32,device='cuda').contiguous()
# volume_shape = torch.tensor([image.shape[1], image.shape[0]], dtype=torch.int32).cuda()
# volume_origin = torch.tensor([[-(image.shape[1]-1)/2, -(image.shape[0]-1)/2],[-(image.shape[1]-1)/2, -(image.shape[0]-1)/2],[-(image.shape[1]-1)/2, -(image.shape[0]-1)/2]], dtype=torch.float32).cuda()
# detector_origin = torch.tensor([[-(detector_width-1) / 2],[-(detector_width-1) / 2],[-(detector_width-1) / 2]], dtype=torch.float32).cuda()
# detector_spacing = torch.tensor([[1.],[1.],[1.]], dtype=torch.float32).cuda()
# sinogram_shape = torch.tensor([number_of_projections,detector_width], dtype=torch.int32).cuda()
# volume_spacing = torch.tensor([[1, 1],[1, 1],[1, 1]], dtype=torch.float32).cuda()
# traj = circular_trajectory_2d_test(number_of_projections, angular_range)
# ray_vectors = torch.tensor([traj,traj,traj], dtype=torch.float32).cuda()

# geometry = get_geometry(volume_shape, volume_origin, volume_spacing, sinogram_shape, detector_origin, detector_spacing,ray_vectors )

# volume_shape =[image.shape[1], image.shape[0]]
# volume_shape =[image.shape[2], image.shape[1], image.shape[0]]

# volume_origin = [[-(image.shape[1]-1)/2, -(image.shape[0]-1)/2],[-(image.shape[1]-1)/2, -(image.shape[0]-1)/2],[-(image.shape[1]-1)/2, -(image.shape[0]-1)/2]]
# detector_origin = [[-(detector_width-1) / 2],[-(detector_width-1) / 2],[-(detector_width-1) / 2]]
# detector_spacing = [1.]
# detector_spacing = [1., 1.]
# sinogram_shape = [number_of_projections, detector_width]
# sinogram_shape = [number_of_projections, detector_height, detector_width]
# volume_spacing = [1, 1]
# volume_spacing = [1, 1, 1]

# geometry = Geometry()
# geometry.init_from_EZRT_header(raw2py.raw2py(r'C:\Users\yzhou\Desktop\data\erdnuss\Erdnuss_meanBPM\erdnuss_0000.raw', switch_order=True))
# geometry.parameter_dict['volume_shape'] = np.asarray([512,1023,1023],dtype=np.float32)
# geometry.parameter_dict['volume_spacing'] = np.asarray([0.0166772,0.0185576,0.0185576],dtype=np.float32)
# geometry.parameter_dict['volume_origin'] = -(geometry.parameter_dict['volume_shape'] - 1) / 2.0 * geometry.parameter_dict['volume_spacing']
# geometry.set_detector_shift([0,10])
# geometry.parameter_dict['detector_origin'] = geometry.parameter_dict['detector_origin'] + ([+10*0.0478516,0])
# geometry.parameter_dict['trajectory'] = circular_trajectory_3d(True,**geometry.parameter_dict)
# geometry.init_from_parameters(volume_shape=volume_shape,volume_spacing=volume_spacing,
#                                 detector_shape=detector_shape,detector_spacing=detector_spacing,
#                                 number_of_projections=number_of_projections,angular_range=angular_range,
#                                 trajectory=circular_trajectory_3d, source_isocenter_distance=950, source_detector_distance=1200)

# geometry.to_json('test_geometry.json')
# test = Geometry.from_json('test_geometry.json')

# input_data = ParallelProjection2D().forward(volume,**geometry)
# input_data = FanProjection2D().forward(volume,**geometry)
# input_data = ConeProjection3D().forward(volume,**geometry)

import os
files = []
data_path = r'C:\Users\yzhou\Desktop\data\erdnuss\Erdnuss_meanBPM'
headers = []
for name in os.listdir(data_path):
    if name.endswith(".raw"):
        files.append(name)
        # header, _ = raw2py.raw2py(os.path.join(data_path,files[0]))

projection_stack = []
for file in files:
    header, raw = raw2py.raw2py(os.path.join(data_path, file))
    headers.append(header)
    projection_stack.append(raw)
geometry = Geometry()
geometry.init_from_EZRT_header(headers)
projection_stack = np.asarray(projection_stack)
inull = 32282
# inull = header.inull_value
projection_stack[:,0:5,0:60] = inull
projection_stack = np.maximum(0, np.log((projection_stack+1e-07)/inull) * -1).astype(np.float32)
# data = raw2py('C:/data/Erdnuss_meanBPM/erdnuss_0000.raw')
pad=12
projection_stack_padded = torch.constant_pad_nd(torch.tensor(projection_stack),(0,0,0,0,0,pad),0)
projection_stack_padded = torch.unsqueeze(projection_stack_padded,0)

reco_filter = torch.tensor(shepp_logan_3D(geometry.detector_shape, geometry.detector_spacing, geometry.number_of_projections+pad),dtype=torch.float32)
x = torch.fft.fft(projection_stack_padded,dim=-1,norm="ortho")
x = torch.multiply(x,reco_filter)
x = torch.fft.ifft(x,dim=-1,norm="ortho").real
x = x[:,0:500,:,:]
reco = ConeBackProjection3D().forward(x.cuda().contiguous(), **geometry)
reco = reco.cpu().numpy()

model = Reconstruction(projection_stack.shape)
# lossf = torch.nn.MSELoss()
lossf = kornia.losses.TotalVariation()
optimizer = torch.optim.Adam(model.parameters())
# geometry.parameter_dict['volume_shape'] = [512,1024,1024]
for epoch in range(10):
    model.train()
    pred = model.forward(torch.Tensor(projection_stack_padded).contiguous(), **geometry)
    # pred = model.forward(torch.Tensor(input_data.cpu().numpy()).cuda().contiguous(), **geometry)

    loss = lossf.forward(pred.cpu())
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# import pyconrad as pyc
# pyc.setup_pyconrad()
#
# pyc.imshow(input_data,'input_data')
# pyc.imshow(pred,'pred')
# a=5