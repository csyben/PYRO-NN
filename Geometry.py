import numpy as np
import tensorflow as tf
import pyconrad as pyc # TODO: get independent of pyconrad


class Geometry:


    def __init__(self, volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance):
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
        # Volume Parameters:
        self.volume_shape = np.array(volume_shape, dtype=np.float32)
        self.volume_spacing = np.array(volume_spacing, dtype=np.float32) # TODO: build in checks for only passing a single number to use in all dims
        self.volume_origin = -(self.volume_shape - 1) / 2.0 * self.volume_spacing

        # Detector Parameters:
        self.detector_shape = np.array(detector_shape, dtype=np.float32)
        self.detector_spacing = np.array(detector_spacing, dtype=np.float32) # TODO: build in checks for only passing a single number to use in all dims
        self.detector_origin = -(self.detector_shape - 1) / 2.0 * self.detector_spacing

        # Trajectory Parameters:
        self.number_of_projections = number_of_projections
        self.angular_range = angular_range
        self.sinogram_shape = np.array([self.number_of_projections, *self.detector_shape], dtype=np.float32)

        self.source_detector_distance = source_detector_distance
        self.source_isocenter_distance = source_isocenter_distance

        # Tensor Protos:
        self.init_tensor_proto_members()


    def init_tensor_proto_members(self): #TODO: better name tensor_proto_ = tp_ ?
        self.tensor_proto_volume_shape = tf.contrib.util.make_tensor_proto(self.volume_shape, tf.float32)
        self.tensor_proto_volume_spacing = tf.contrib.util.make_tensor_proto(self.volume_spacing, tf.float32)
        self.tensor_proto_volume_origin = tf.contrib.util.make_tensor_proto(self.volume_origin, tf.float32)

        self.tensor_proto_detector_shape = tf.contrib.util.make_tensor_proto(self.detector_shape, tf.float32)
        self.tensor_proto_detector_spacing = tf.contrib.util.make_tensor_proto(self.detector_spacing, tf.float32)
        self.tensor_proto_detector_origin = tf.contrib.util.make_tensor_proto(self.detector_origin, tf.float32)

        self.tensor_proto_sinogram_shape = tf.contrib.util.make_tensor_proto(self.sinogram_shape, tf.float32)


    def to_tensor_proto(self, member): 
        return tf.contrib.util.make_tensor_proto(member, tf.float32)


class Geometry_2d(Geometry):


    def init_trajectory(self, number_of_projections, angular_range):
        rays = np.zeros([number_of_projections, 2])
        angular_increment = angular_range / number_of_projections
        for i in np.arange(0, number_of_projections):
            rays[i] = [np.cos(i * angular_increment), np.sin(i * angular_increment)]
        return rays


    def __init__(self, volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance):

        # init base Geometry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing, [detector_shape], [detector_spacing], number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)

        # init class specific members
        self.ray_vectors = self.init_trajectory(number_of_projections, angular_range)
        self.tensor_proto_ray_vectors = tf.contrib.util.make_tensor_proto(self.ray_vectors, tf.float32)


class Geometry_3d(Geometry):


    #TODO: here the projection matrices(3x4) from conrad go in. TODO: Get the necessary stuff from conrad.
    def init_trajectory(self):

        _projection_matrix = np.zeros((self.number_of_projections, 3, 4))

        pyc.setup_pyconrad()
        pyc.start_gui()

        _ = pyc.ClassGetter('edu.stanford.rsl.conrad.geometry.trajectories','edu.stanford.rsl.conrad.geometry')

        #circ_traj = pyc.edu().stanford.rsl.conrad.geometry.trajectories.CircularTrajectory()

        circ_traj = _.CircularTrajectory()
        circ_traj.setSourceToDetectorDistance(self.source_detector_distance)

        circ_traj.setPixelDimensionX(np.float64(self.detector_spacing[0]))
        circ_traj.setPixelDimensionY(np.float64(self.detector_spacing[1]))
        circ_traj.setDetectorHeight(int(self.detector_shape[0]))
        circ_traj.setDetectorWidth(int(self.detector_shape[1]))

        circ_traj.setOriginInPixelsX(np.float64(self.volume_origin[2]))
        circ_traj.setOriginInPixelsY(np.float64(self.volume_origin[1]))
        circ_traj.setOriginInPixelsZ(np.float64(self.volume_origin[0]))
        circ_traj.setReconDimensions(np.flip(self.volume_shape).tolist())
        circ_traj.setReconVoxelSizes(np.flip(self.volume_spacing).tolist())

        DETECTORMOTION_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 1)
        ROTATIONAXIS_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 3 )

        average_angular_increment = 1.0
        detector_offset_u = 0
        detector_offset_v = 0
        rotationAxis = _.SimpleVector.from_list([0,0,1])
        center = _.PointND.from_list([0,0,0])

        circ_traj.setTrajectory(self.number_of_projections, self.source_detector_distance, average_angular_increment,
                                detector_offset_u, detector_offset_v, DETECTORMOTION_MINUS, ROTATIONAXIS_MINUS, rotationAxis, center, 0)


        for proj in range(0, self.number_of_projections):
            _projection_matrix[proj] = circ_traj.getProjectionMatrix(proj).computeP().as_numpy()

        return _projection_matrix


    def __init__(self, volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance):

        # init base Geometry class with 3 dimensional members:
        super().__init__(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)

        # init class specific members
        self.projection_matrices = self.init_trajectory()
        self.tensor_proto_projection_matrices = tf.contrib.util.make_tensor_proto(self.projection_matrices, tf.float32)


# TODO: are these even a good idea? yes but make dict
def get_parallel_projection2d_params(geo):
    return (geo.volume_shape, geo.sinogram_shape, geo.tensor_proto_volume_origin, geo.tensor_proto_detector_origin, geo.tensor_proto_volume_spacing, geo.tensor_proto_detector_spacing, geo.tensor_proto_ray_vectors)

# TODO: why are not all params a tensor proto?
def get_parallel_backprojection2d_params(geo):
    return (geo.sinogram_shape, geo.volume_shape, geo.tensor_proto_volume_origin, geo.tensor_proto_detector_origin, geo.tensor_proto_volume_spacing, geo.tensor_proto_detector_spacing, geo.tensor_proto_ray_vectors)