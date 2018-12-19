import numpy as np
from .geometry_base import GeometryBase


class GeometryParallel2D(GeometryBase):
    """
        2D Parallel specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range,
                 trajectory_defining_function = None):

        # init base selfmetry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         [detector_shape], [detector_spacing],
                         number_of_projections, angular_range,
                         None, None)

        # init class specific members
        if trajectory_defining_function is not None:
            self.ray_vectors = trajectory_defining_function(self)
            self.tensor_proto_ray_vectors = super().to_tensor_proto(self.ray_vectors)

    def set_ray_vectors(self, ray_vectors):
        """
            Sets the member ray_vectors.
        Args:
            ray_vectors: np.array defining the trajectory ray_vectors.
        """
        self.ray_vectors = np.array(ray_vectors, self.np_dtype)
        self.tensor_proto_ray_vectors = super().to_tensor_proto(self.ray_vectors)

    def get_parallel_projection2d_params_dict(self):
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
                "ray_vectors":                  self.tensor_proto_ray_vectors}

    def get_parallel_backprojection2d_params_dict(self):
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
                "ray_vectors":                  self.tensor_proto_ray_vectors}
