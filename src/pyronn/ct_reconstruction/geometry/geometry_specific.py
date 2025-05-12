import numpy as np
from abc import ABC, abstractmethod
import pyronn
pyronn.set_backend('torch')

from ..layers.projection_3d import ConeProjectionFor3D
from .geometry_base import GeometryCone3D
from ..helpers.phantoms import shepp_logan
from ..helpers.phantoms.primitives_3d import generate_3D_primitives
from ..helpers.trajectories.circular_trajectory import circular_trajectory_3d
from ..helpers.trajectories.arbitrary_trajectory import arbitrary_projection_matrix

class SpecificGeometry(ABC):
    def __init__(self, geo_info_dict, traj_func):
        """
        Generate a specific geometry
        Args:
            geo_info_dict: All required information for creating a geometry.
        """
        self.geometry_info = geo_info_dict
        self.geometry = self.set_geo()
        temp_info = {**geo_info_dict, **self.geometry.get_dict()}
        self.trajectory = traj_func(**temp_info)
        self.geometry.set_trajectory(self.trajectory)
    @abstractmethod
    def set_geo(self): pass

    @abstractmethod
    def create_sinogram(self, phantom): pass
    def generate_specific_phantom(self, phantom_func, *args, **kwargs):
        """
        Generates a phantom created by the given function and its corresponding sinogram.

        The method first creates a phantom based on the volume shape specified in the
        geometry attribute of the class. It then computes the sinogram by applying a forward projection.
        The projection is calculated based on the parameters defined in the
        geometry attribute, including the detector shape, spacing, and the source-detector configuration.

        Returns:
            Tuple[np.array, np.array]: A tuple containing two numpy arrays. The first array is the generated
            3D mask of the phantom, and the second array is the corresponding 3D sinogram obtained through
            the cone beam forward projection.
        """
        phantom = phantom_func(self.geometry_info['volume_shape'], *args, **kwargs)
        phantom = np.expand_dims(phantom, axis=0)
        mask = (phantom != 0)
        sinogram = self.create_sinogram(phantom)
        return mask, sinogram, phantom, self.geometry


class CircularGeometrys3D(SpecificGeometry):
    def __init__(self, geo_dict_info):
        super().__init__(geo_dict_info, circular_trajectory_3d)

    def set_geo(self):
        geometry = GeometryCone3D(**self.geometry_info)
        return geometry

    def create_sinogram(self, phantom):
        return ConeProjectionFor3D().forward(phantom, self.geometry)

class ArbitraryGeometrys3D(SpecificGeometry):
    def __init__(self, geo_dict_info):
        super().__init__(geo_dict_info, arbitrary_projection_matrix)

    def set_geo(self):
        geometry = GeometryCone3D(**self.geometry_info)
        return geometry
    def create_sinogram(self, phantom):
        return ConeProjectionFor3D().forward(phantom, self.geometry)


if __name__ == '__main__':
    geo_dict_info = {'volume_shape': [256, 256, 256],
                     'volume_spacing': [0.5, 0.5, 0.5],
                     'detector_shape': [400, 600],
                     'detector_spacing': [1, 1],
                     'number_of_projections': 360,
                     'angular_range': 2*np.pi,
                     'source_isocenter_distance': 750,
                     'source_detector_distance': 1200,
                     'swap_detector_axis': False
                     }
    cg = CircularGeometrys3D(geo_dict_info)
    mask, sino, geo = cg.generate_specific_phantom(shepp_logan.shepp_logan_3d)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(mask[0][125], cmap='gray')
    plt.savefig('mask.png')
    plt.imshow(sino[0][125], cmap='gray')
    plt.savefig('sino.png')