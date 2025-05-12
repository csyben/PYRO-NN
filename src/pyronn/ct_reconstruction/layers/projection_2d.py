import pyronn
import numpy as np

class ParallelProjectionFor2D:
    def forward(self, input, geometry, for_train=False, debug=False):
        '''
        Projection for the 2D parallel beam CT.

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
            from pyronn.ct_reconstruction.layers.torch.projection_2d import ParallelProjection2D

            if not isinstance(input, torch.Tensor):
                phantom = torch.tensor(input.copy(), dtype=torch.float32)
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
                except Exception as e:
                    if isinstance(e, TypeError):
                        if debug: print('Attribute <' + k + '> could not be transformed to torch.Tensor')
                    else:
                        raise e
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
    def forward(self, input, geometry, for_train=False, debug=False):
        '''
        Projection for the 2D fan beam CT.

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
            from pyronn.ct_reconstruction.layers.torch.projection_2d import FanProjection2D

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
                except Exception as e:
                    if isinstance(e, TypeError):
                        if debug: print('Attribute <' + k + '> could not be transformed to torch.Tensor')
                    else:
                        raise e

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