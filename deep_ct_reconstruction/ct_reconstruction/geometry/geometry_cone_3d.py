import numpy as np
from .geometry_base import GeometryBase


class GeometryCone3D(GeometryBase):
    """
        3D Cone specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range,
                 source_detector_distance, source_isocenter_distance,
                 trajectory_defining_function = None):

        # init base Geometry class with 3 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         detector_shape, detector_spacing,
                         number_of_projections, angular_range,
                         source_detector_distance, source_isocenter_distance)

        # init class specific members
        if trajectory_defining_function is not None:
            self.projection_matrices = trajectory_defining_function(self)
            self.tensor_proto_projection_matrices = super().to_tensor_proto(self.projection_matrices)

    def set_projection_matrices(self, projection_matrices):
        """
            Sets the member projection_matrices.
        Args:
            projection_matrices: np.array defining the trajectory projection_matrices.
        """
        self.projection_matrices = np.array(projection_matrices, self.np_dtype)
        self.tensor_proto_projection_matrices = super().to_tensor_proto(self.projection_matrices)

    def get_cone_projection3d_params_dict(self, hardware_interp=False, step_size=1.0):
        """
        Convenience function for making the layer call easier. The named parameters correspond to those in the
        cuda code.
        Returns:
             Dict that has to be dereferenced with the * operator in the layer call.
        """
        return {"volume_shape":                 self.volume_shape,
                "projection_shape":             self.sinogram_shape,
                "volume_origin":                self.tensor_proto_volume_origin,
                "volume_spacing":               self.tensor_proto_volume_spacing,
                "projection_matrices":          self.tensor_proto_projection_matrices,
                "hardware_interp":              hardware_interp,
                "step_size":                    step_size}

    def get_cone_backprojection3d_params_dict(self, hardware_interp=False):
        """
        Convenience function for making the layer call easier. The named parameters correspond to those in the
        cuda code.
        Returns:
             Dict that has to be dereferenced with the * operator in the layer call.
        """
        return {"sinogram_shape":               self.sinogram_shape,
                "volume_shape":                 self.volume_shape,
                "volume_origin":                self.tensor_proto_volume_origin,
                "volume_spacing":               self.tensor_proto_volume_spacing,
                "projection_matrices":          self.tensor_proto_projection_matrices,
                "hardware_interp":              hardware_interp}
