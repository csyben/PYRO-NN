import pyronn
import numpy as np


class Projection3D:
    def __init__(self):
        self.backend = pyronn.read_backend()

    def forward(self):
        pass

class ConeProjectionFor3D(Projection3D):
    def forward(self, input, geometry, for_train=False):
        '''
        Projection for the 3D cone beam CT.

        args:
            input: (1, number_of_projections, detection_size) numpy array or torch.Tensor.
            geometry: The projection geometry used for projection.
            for_train: Set the return value data type if the backend is torch. You can get a numpy.array by setting this
            value False, otherwise you will get a torch.Tensor.

        return:
            The reconstruction result of 3D cone beam CT.
        '''
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.projection_3d import ConeProjection3D
            print('work on torch')

            if not isinstance(input, torch.Tensor):
                phantom = torch.tensor(input.copy(), dtype=torch.float32).cuda()
            else:
                phantom = torch.clone(input).cuda()

            tensor_geometry = {}
            geo_dict = vars(geometry)
            for k in geo_dict:
                param = geo_dict[k]
                try:
                    if hasattr(param, '__len__'):
                        tmp_tensor = torch.Tensor(param)
                    else:
                        tmp_tensor = torch.Tensor([param])


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