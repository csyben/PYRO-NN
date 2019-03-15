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
            source_detector_distance:   The source to detector distance (sdd).
            source_isocenter_distance:  The source to isocenter distance (sid).
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

    def to_tensor_proto(self, member):
        """
            Wraps the TensorProto creation function.
        """
        return tf.contrib.util.make_tensor_proto(member, tf.float32)

    class SetTensorProtoProperty(object):
        """
            A setter descriptor that only intercepts variable assignment.
            Allows for setting of the corresponding TensorProto members.
            Use @SetTensorProtoProperty for every member that also needs TensorProto members.
        """

        def __init__(self, func, doc=None):
            self.func = func
            self.__doc__ = doc if doc is not None else func.__doc__

        def __set__(self, obj, value):
            return self.func(obj, value)

    @SetTensorProtoProperty
    def volume_shape(self, value):
        self.__dict__['volume_shape'] = value
        self.tensor_proto_volume_shape = self.to_tensor_proto(self.volume_shape)

    @SetTensorProtoProperty
    def volume_spacing(self, value):
        self.__dict__['volume_spacing'] = value
        self.tensor_proto_volume_spacing = self.to_tensor_proto(self.volume_spacing)

    @SetTensorProtoProperty
    def volume_origin(self, value):
        self.__dict__['volume_origin'] = value
        self.tensor_proto_volume_origin = self.to_tensor_proto(self.volume_origin)

    @SetTensorProtoProperty
    def detector_shape(self, value):
        self.__dict__['detector_shape'] = value
        self.tensor_proto_detector_shape = self.to_tensor_proto(self.detector_shape)

    @SetTensorProtoProperty
    def detector_spacing(self, value):
        self.__dict__['detector_spacing'] = value
        self.tensor_proto_detector_spacing = self.to_tensor_proto(self.detector_spacing)

    @SetTensorProtoProperty
    def detector_origin(self, value):
        self.__dict__['detector_origin'] = value
        self.tensor_proto_detector_origin = self.to_tensor_proto(self.detector_origin)

    @SetTensorProtoProperty
    def sinogram_shape(self, value):
        self.__dict__['sinogram_shape'] = value
        self.tensor_proto_sinogram_shape = self.to_tensor_proto(self.sinogram_shape)
