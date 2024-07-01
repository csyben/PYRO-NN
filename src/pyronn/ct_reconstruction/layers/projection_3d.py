import pyronn
import numpy as np


class Projection3D:
    def __init__(self):
        self.backend = pyronn.read_backend()

    def forward(self):
        pass

class ConeProjectionFor3D(Projection3D):
    def forward(self, input, geometry, for_train=False):
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.projection_3d import ConeProjection3D
            print('work on torch')

            phantom = torch.tensor(input.copy(), dtype=torch.float32)

            tensor_geometry = {}
            geo_dict = vars(geometry)
            for k in geo_dict:
                param = geo_dict[k]
                try:
                    if hasattr(param, '__len__'):
                        tmp_tensor = torch.Tensor(param)
                    else:
                        tmp_tensor = torch.Tensor([param])

                    phantom = torch.tensor(input.copy(), dtype=torch.float32).cuda()
                    tensor_geometry[k] = tmp_tensor.cuda()
                except:
                    print('Attribute <' + k + '> could not be transformed to torch.Tensor')

            sinogram = ConeProjection3D().forward(phantom, **tensor_geometry)
            if for_train:
                return sinogram

            if sinogram.device.type == 'cuda':
                return sinogram.detach().cpu().numpy()
            return sinogram.cpu().numpy()

        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                from pyronn.ct_reconstruction.layers.tensorflow.projection_3d import cone_projection3d
                return cone_projection3d(input, geometry)
            else:
                raise e