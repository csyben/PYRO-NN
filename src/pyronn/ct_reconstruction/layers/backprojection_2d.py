import pyronn
import numpy as np


class ParallelBackProjectionFor2D:
    def forward(self, input, geometry, for_train=False):
        '''
        Reconstruction for the 2D parallel beam CT.

        args:
            input: (1, number_of_projections, detection_size) numpy array or torch.Tensor.
            geometry: The projection geometry used for projection.
            for_train: Set the return value data type if the backend is torch. You can get a numpy.array by setting this
            value False, otherwise you will get a torch.Tensor.

        return:
            The reconstruction result of 2D parallel beam CT.
        '''
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.backprojection_2d import ParallelBackProjection2D
            print('work on torch')

            if not isinstance(input, torch.Tensor):
                sinogram = torch.tensor(input.copy(), dtype=torch.float32)
            else:
                sinogram = torch.clone(input)

            tensor_geometry = {}
            geo_dict = vars(geometry)
            for k in geo_dict:
                param = geo_dict[k]
                try:
                    if hasattr(param, '__len__'):
                        tmp_tensor = torch.Tensor(param)

                        sinogram = sinogram.cuda()
                        tensor_geometry[k] = tmp_tensor.cuda()
                except:
                    print('Attribute <' + k + '> could not be transformed to torch.Tensor')

            reco = ParallelBackProjection2D().forward(sinogram.contiguous(), **tensor_geometry)
            if for_train:
                return reco

            if reco.device.type == 'cuda':
                return reco.detach().cpu().numpy()
            return reco.cpu().numpy()

        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                from pyronn.ct_reconstruction.layers.tensorflow.backprojection_2d import parallel_backprojection2d
                return parallel_backprojection2d(input, geometry)
            else:
                raise e

class FanBackProjectionFor2D:
    def forward(self, input, geometry, for_train=False):
        '''
        Reconstruction for the 2D fan beam CT.

        args:
            input: (1, number_of_projections, detection_size) numpy array or torch.Tensor.
            geometry: The projection geometry used for projection.
            for_train: Set the return value data type if the backend is torch. You can get a numpy.array by setting this
            value False, otherwise you will get a torch.Tensor.

        return:
            The reconstruction result of 2D fan beam CT.
        '''
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.backprojection_2d import FanBackProjection2D
            print('work on torch')

            if not isinstance(input, torch.Tensor):
                sinogram = torch.tensor(input.copy(), dtype=torch.float32).cuda()
            else:
                sinogram = torch.clone(input).cuda()

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
            reco = FanBackProjection2D().forward(sinogram.contiguous(), **tensor_geometry)
            if for_train:
                return reco

            if reco.device.type == 'cuda':
                return reco.detach().cpu().numpy()
            return reco.cpu().numpy()

        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                from pyronn.ct_reconstruction.layers.tensorflow.backprojection_2d import fan_backprojection2d
                return fan_backprojection2d(input, geometry)
            else:
                raise e

