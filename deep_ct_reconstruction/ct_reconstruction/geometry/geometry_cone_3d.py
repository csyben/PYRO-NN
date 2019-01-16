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


    def set_projection_matrices(self, projection_matrices):
        """
            Sets the member projection_matrices.
        Args:
            projection_matrices: np.array defining the trajectory projection_matrices.
        """
        self.projection_matrices = np.array(projection_matrices, self.np_dtype)
        self.tensor_proto_projection_matrices = super().to_tensor_proto(self.projection_matrices)
