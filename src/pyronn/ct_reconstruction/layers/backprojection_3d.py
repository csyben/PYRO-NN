import pyronn
import numpy as np

class ConeBackProjectionFor3D:
    def forward(self, input, geometry, for_train=False):
        try:
            import torch
            from pyronn.ct_reconstruction.layers.torch.backprojection_3d import ConeBackProjection3D
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
                    else:
                        tmp_tensor = torch.Tensor([param])

                    sinogram = sinogram.cuda()
                    tensor_geometry[k] = tmp_tensor.cuda()
                except:
                    print('Attribute <' + k + '> could not be transformed to torch.Tensor')
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