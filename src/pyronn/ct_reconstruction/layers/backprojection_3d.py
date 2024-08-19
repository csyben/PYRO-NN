import pyronn
import numpy as np

class ConeBackProjectionFor3D:
    def forward(self, input, geometry, for_train=False, debug=False):
        '''
        Reconstruction for the 3D cone beam CT.

        args:
            input: (1, number_of_projections, detector_height, detector_width) numpy array or torch.Tensor.
            geometry: The projection geometry used for projection.
            for_train: Set the return value data type if the backend is torch. You can get a numpy.array by setting this
            value False, otherwise you will get a torch.Tensor.

        return:
            The reconstruction result of 2D parallel beam CT.
        '''
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.backprojection_3d import ConeBackProjection3D

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
                except Exception as e:
                    if isinstance(e, TypeError):
                        if debug: print('Attribute <' + k + '> could not be transformed to torch.Tensor')
                    else:
                        raise e

            reco = ConeBackProjection3D().forward(sinogram.contiguous(), **tensor_geometry)
            if for_train:
                return reco

            if reco.device.type == 'cuda':
                return reco.detach().cpu().numpy()
            return reco.cpu().numpy()

        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                from pyronn.ct_reconstruction.layers.tensorflow.backprojection_3d import cone_backprojection3d
                return cone_backprojection3d(input, geometry)
            else:
                raise e