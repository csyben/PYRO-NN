import copy
import numpy as np
# import pyronn
# BACKEND = pyronn.read_backend()

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
                 source_isocenter_distance,
                 *args, **kwargs):
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
        # self.gpu_device = True
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
        if isinstance(angular_range, list):
            self.angular_range = angular_range
        else:
            self.angular_range = [0, angular_range]

        self.sinogram_shape = np.array([self.number_of_projections, *self.detector_shape])

        self.source_detector_distance = source_detector_distance
        self.source_isocenter_distance = source_isocenter_distance
        self.fan_angle = None
        self.cone_angle = None
        self.projection_multiplier = None
        self.step_size = None


    # def cuda(self):
    #     self.gpu_device = True
    #
    # def cpu(self):
    #     self.gpu_device = False

    def set_trajectory(self, trajectory):
        """
            Sets the member trajectory.
        Args:
            trajectory: np.array defining the trajectory.
        """
        self.trajectory = np.array(trajectory, self.np_dtype)

    def update(self, dict):
        changed = []
        for key in dict:
            if key in dir(self):
                setattr(self, key, dict[key])
                self.key = dict[key]
                changed.append(key)
            else:
                print(f'{key} is not a property of geometry! Please check it!')
        if changed:
            print(f'The following properties has been changed: {changed}')
            if 'trajectory' not in changed:
                print(f'Please confirm whether you need to modify the trajectory.')

    def get_dict(self):
        info = {}
        for i in dir(self):
            if i[:2] != '__': info[i] = getattr(self, i)
        return info


class GeometryParallel2D(GeometryBase):
    """
        2D Parallel specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range, *args, **kwargs):
        # init base selfmetry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         detector_shape, detector_spacing,
                         number_of_projections, angular_range,
                         None, None, *args, **kwargs)


class GeometryFan2D(GeometryBase):
    """
        2D Fan specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range,
                 source_detector_distance, source_isocenter_distance, *args, **kwargs):
        # init base Geometry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         detector_shape, detector_spacing,
                         number_of_projections, angular_range,
                         source_detector_distance, source_isocenter_distance, *args, **kwargs)

        # defined by geometry so calculate for convenience use
        self.fan_angle = np.arctan(((self.detector_shape[0] - 1) / 2.0 * self.detector_spacing[0]) / self.source_detector_distance)


class GeometryCone3D(GeometryBase):
    """
        3D Cone specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, angular_range,
                 source_detector_distance, source_isocenter_distance, *args, **kwargs):
        # init base Geometry class with 3 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         detector_shape, detector_spacing,
                         number_of_projections, angular_range,
                         source_detector_distance, source_isocenter_distance, *args, **kwargs)

        # defined by geometry so calculate for convenience use
        self.fan_angle = np.arctan(((self.detector_shape[1] - 1) / 2.0 * self.detector_spacing[1]) / self.source_detector_distance)
        self.cone_angle = np.arctan(((self.detector_shape[0] - 1) / 2.0 * self.detector_spacing[0]) / self.source_detector_distance)

        # Containing the constant part of the distance weight and discretization invariant
        self.projection_multiplier = self.source_isocenter_distance * self.source_detector_distance * detector_spacing[-1] * np.pi / self.number_of_projections
        # TODO: need to be changed or not?
        self.step_size = 0.2
