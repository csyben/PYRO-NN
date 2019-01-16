import numpy as np
import tensorflow as tf


class GeometryBase:
    """
        The Base Class for the different Geometry classes. Provides commonly used members.
    """

    def __init__(self,
                 volume_shape,
                 volume_spacing,
                 detector_shape,
                 detector_spacing,
                 number_of_projections,
                 angular_range,
                 source_detector_distance,
                 source_isocenter_distance):
        """
            Constructor of Base Geometry Class, should only get called by sub classes.
        Args:
            volume_shape:               The volume size in Z, Y, X order.
            volume_spacing:             The spacing between voxels in Z, Y, X order.
            detector_shape:             Shape of the detector in Y, X order.
            detector_spacing:           The spacing between detector voxels in Y, X order.
            number_of_projections:      Number of equidistant projections.
            angular_range:              The covered angular range.
            source_detector_distance:   ...
            source_isocenter_distance:  ...
        """
        self.np_dtype = np.float32  # datatype for np.arrays make sure everything will be float32

        # Volume Parameters:
        self.volume_shape = np.array(volume_shape)
        self.volume_spacing = np.array(volume_spacing, dtype=self.np_dtype)
        self.volume_origin = -(self.volume_shape - 1) / 2.0 * self.volume_spacing

        # Detector Parameters:
        self.detector_shape = np.array(detector_shape)
        self.detector_spacing = np.array(detector_spacing, dtype=self.np_dtype)
        self.detector_origin = -(self.detector_shape - 1) / 2.0 * self.detector_spacing

        # Trajectory Parameters:
        self.number_of_projections = number_of_projections
        self.angular_range = angular_range
        self.sinogram_shape = np.array([self.number_of_projections, *self.detector_shape])

        self.source_detector_distance = source_detector_distance
        self.source_isocenter_distance = source_isocenter_distance

        # Tensor Protos:
        self.init_tensor_proto_members()

    def init_tensor_proto_members(self):
        """
            This function inits all members again as tensor_proto members for the layer calls.
        """
        self.tensor_proto_volume_shape = self.to_tensor_proto(self.volume_shape)
        self.tensor_proto_volume_spacing = self.to_tensor_proto(self.volume_spacing)
        self.tensor_proto_volume_origin = self.to_tensor_proto(self.volume_origin)

        self.tensor_proto_detector_shape = self.to_tensor_proto(self.detector_shape)
        self.tensor_proto_detector_spacing = self.to_tensor_proto(self.detector_spacing)
        self.tensor_proto_detector_origin = self.to_tensor_proto(self.detector_origin)

        self.tensor_proto_sinogram_shape = self.to_tensor_proto(self.sinogram_shape)

    def to_tensor_proto(self, member):
        """
            Wraps the TensorProto creation function.
        """
        return tf.contrib.util.make_tensor_proto(member, tf.float32)
