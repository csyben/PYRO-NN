import numpy as np
from .geometry_base import GeometryBase


class GeometryParallel2D(GeometryBase):
    """
        2D Parallel specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range):
        # init base selfmetry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         [detector_shape], [detector_spacing],
                         number_of_projections, angular_range,
                         None, None)

    def set_ray_vectors(self, ray_vectors):
        """
            Sets the member ray_vectors.
        Args:
            ray_vectors: np.array defining the trajectory ray_vectors.
        """
        self.ray_vectors = np.array(ray_vectors, self.np_dtype)

    @GeometryBase.SetTensorProtoProperty
    def ray_vectors(self, value):
        self.__dict__['ray_vectors'] = value
        self.tensor_proto_ray_vectors = super().to_tensor_proto(self.ray_vectors)
