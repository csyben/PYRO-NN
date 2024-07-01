import pyronn
import numpy as np

class ParallelProjectionFor2D:
    def forward(self, input, geometry, for_train=False):
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.projection_2d import ParallelProjection2D
            print('work on torch')

            phantom = torch.tensor(input.copy(), dtype=torch.float32)
            print(type(phantom))
           
            tensor_geometry = {}
            geo_dict = vars(geometry)
            for k in geo_dict:
                param = geo_dict[k]
                if hasattr(param, '__len__'):
                    tmp_tensor = torch.Tensor(param)

                    phantom = torch.tensor(input.copy(), dtype=torch.float32).cuda()
                    tensor_geometry[k] = tmp_tensor.cuda()
            sinogram =  ParallelProjection2D().forward(phantom, **tensor_geometry)
            if for_train:
                return sinogram

            if sinogram.device.type == 'cuda':
                return sinogram.detach().cpu().numpy()
            return sinogram.cpu().numpy()

        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                from pyronn.ct_reconstruction.layers.tensorflow.projection_2d import parallel_projection2d
                return parallel_projection2d(input, geometry)
            else:
                raise e


class FanProjectionFor2D:
    def forward(self, input, geometry, for_train=False):
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.projection_2d import FanProjection2D
            print('work on torch')

            phantom = torch.tensor(input.copy(), dtype=torch.float32)

            tensor_geometry = {}
            geo_dict = vars(geometry)
            for k in geo_dict:
                param = geo_dict[k]
                try :
                    if hasattr(param, '__len__'):
                        tmp_tensor = torch.Tensor(param)
                    else:
                        tmp_tensor = torch.Tensor([param])

                    phantom = torch.tensor(input.copy(), dtype=torch.float32).cuda()
                    tensor_geometry[k] = tmp_tensor.cuda()
                except:
                    print('Attribute <' + k + '> could not be transformed to torch.Tensor')

            sinogram = FanProjection2D().forward(phantom, **tensor_geometry)
            if for_train:
                return sinogram

            if sinogram.device.type == 'cuda':
                return sinogram.detach().cpu().numpy()
            return sinogram.cpu().numpy()

        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                from pyronn.ct_reconstruction.layers.tensorflow.projection_2d import fan_projection2d
                return fan_projection2d(input, geometry)
            else: raise e