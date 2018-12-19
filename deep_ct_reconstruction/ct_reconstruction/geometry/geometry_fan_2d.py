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
                 source_detector_distance, source_isocenter_distance,
                 trajectory_defining_function = None):

        # init base Geometry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         [detector_shape], [detector_spacing],
                         number_of_projections, angular_range,
                         source_detector_distance, source_isocenter_distance)

        # init class specific members
        if trajectory_defining_function is not None:
            self.central_ray_vectors = trajectory_defining_function(self)
            self.tensor_proto_central_ray_vectors = super().to_tensor_proto(self.central_ray_vectors)

    def set_central_ray_vectors(self, central_ray_vectors):
        """
            Sets the member central_ray_vectors.
        Args:
            central_ray_vectors: np.array defining the trajectory central_ray_vectors.
        """
        self.central_ray_vectors = np.array(central_ray_vectors, self.np_dtype)
        self.tensor_proto_projection_matrices = super().to_tensor_proto(self.central_ray_vectors)

    def get_fan_projection2d_params_dict(self):
        """
        Convenience function for making the layer call easier. The named parameters correspond to those in the
        cuda code.
        Returns:
             Dict that has to be dereferenced with the * operator in the layer call.
        """
        return {"volume_shape":                 self.volume_shape,
                "projection_shape":             self.sinogram_shape,
                "volume_origin":                self.tensor_proto_volume_origin,
                "detector_origin":              self.tensor_proto_detector_origin,
                "volume_spacing":               self.tensor_proto_volume_spacing,
                "detector_spacing":             self.tensor_proto_detector_spacing,
                "source_2_iso_distance":        self.source_isocenter_distance,
                "source_2_detector_distance":   self.source_isocenter_distance,
                "central_ray_vectors":          self.tensor_proto_central_ray_vectors}

    def get_fan_backprojection2d_params_dict(self):
        """
        Convenience function for making the layer call easier. The named parameters correspond to those in the
        cuda code.
        Returns:
             Dict that has to be dereferenced with the * operator in the layer call.
        """
        return {"sinogram_shape":               self.sinogram_shape,
                "volume_shape":                 self.volume_shape,
                "volume_origin":                self.tensor_proto_volume_origin,
                "detector_origin":              self.tensor_proto_detector_origin,
                "volume_spacing":               self.tensor_proto_volume_spacing,
                "detector_spacing":             self.tensor_proto_detector_spacing,
                "source_2_iso_distance":        self.source_isocenter_distance,
                "source_2_detector_distance":   self.source_isocenter_distance,
                "central_ray_vectors":          self.tensor_proto_central_ray_vectors}
