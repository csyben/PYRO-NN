import numpy as np
from .geometry_base import GeometryBase


class GeometryFan2D(GeometryBase):
    """
        2D Fan specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range,
                 source_detector_distance, source_isocenter_distance):
        # init base Geometry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         [detector_shape], [detector_spacing],
                         number_of_projections, angular_range,
                         source_detector_distance, source_isocenter_distance)

        # defined by geometry so calculate for convenience use
        self.fan_angle = np.arctan(((self.detector_shape[0] - 1) / 2.0 * self.detector_spacing[0]) / self.source_detector_distance)

    def set_central_ray_vectors(self, central_ray_vectors):
        """
            Sets the member central_ray_vectors.
        Args:
            central_ray_vectors: np.array defining the trajectory central_ray_vectors.
        """
        self.central_ray_vectors = np.array(central_ray_vectors, self.np_dtype)

    @GeometryBase.SetTensorProtoProperty
    def central_ray_vectors(self, value):
        self.__dict__['central_ray_vectors'] = value
        self.tensor_proto_central_ray_vectors = super().to_tensor_proto(self.central_ray_vectors)
