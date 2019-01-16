import numpy as np
import pyconrad as pyc


def circular_trajectory_2d(geometry):
    """
        Generates the central ray vectors defining a circular trajectory for use with the 2d projection layers.
    Args:
        geometry: 2d Geometry class including angular_range and number_of_projections
    Returns:
        Central ray vectors as np.array.
    """
    rays = np.zeros([geometry.number_of_projections, 2])
    angular_increment = geometry.angular_range / geometry.number_of_projections
    for i in range(geometry.number_of_projections):
        rays[i] = [np.cos(i * angular_increment), np.sin(i * angular_increment)]
    return rays


def circular_trajectory_3d_pyconrad(geometry): # TODO: still have to get independent of pyconrad
    """
        Generates the projection matrices defining a circular trajectory for use with the 3d projection layers.
    Args:
        geometry: 3d Geometry class including angular_range, number_of_projections, source_detector_distance,
        detector_shape, detector_spacing, volume_origin, volume_shape and volume_spacing.
    Returns:
        Projection matrices as np.array.
    """

    _projection_matrix = np.zeros((geometry.number_of_projections, 3, 4))

    pyc.setup_pyconrad()
    pyc.start_gui()

    _ = pyc.ClassGetter('edu.stanford.rsl.conrad.geometry.trajectories', 'edu.stanford.rsl.conrad.geometry')

    # circ_traj = pyc.edu().stanford.rsl.conrad.geometry.trajectories.CircularTrajectory()

    circ_traj = _.CircularTrajectory()
    circ_traj.setSourceToDetectorDistance(geometry.source_detector_distance)

    circ_traj.setPixelDimensionX(np.float64(geometry.detector_spacing[0]))
    circ_traj.setPixelDimensionY(np.float64(geometry.detector_spacing[1]))
    circ_traj.setDetectorHeight(int(geometry.detector_shape[0]))
    circ_traj.setDetectorWidth(int(geometry.detector_shape[1]))

    circ_traj.setOriginInPixelsX(np.float64(geometry.volume_origin[2]))
    circ_traj.setOriginInPixelsY(np.float64(geometry.volume_origin[1]))
    circ_traj.setOriginInPixelsZ(np.float64(geometry.volume_origin[0]))
    circ_traj.setReconDimensions(np.flip(geometry.volume_shape).tolist())
    circ_traj.setReconVoxelSizes(np.flip(geometry.volume_spacing).tolist())

    DETECTORMOTION_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 1)
    ROTATIONAXIS_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 3)

    average_angular_increment = 1.0
    detector_offset_u = 0
    detector_offset_v = 0
    rotationAxis = _.SimpleVector.from_list([0, 0, 1])
    center = _.PointND.from_list([0, 0, 0])

    circ_traj.setTrajectory(geometry.number_of_projections, geometry.source_detector_distance, average_angular_increment,
                            detector_offset_u, detector_offset_v, DETECTORMOTION_MINUS, ROTATIONAXIS_MINUS, rotationAxis, center, 0)

    for proj in range(0, geometry.number_of_projections):
        _projection_matrix[proj] = circ_traj.getProjectionMatrix(proj).computeP().as_numpy()

    return _projection_matrix


def circular_trajectory_3d(geometry):
    """
        Generates the projection matrices defining a circular trajectory for use with the 3d projection layers.
    Args:
        geometry: 3d Geometry class including angular_range, number_of_projections, source_detector_distance,
        detector_shape, detector_spacing, volume_origin, volume_shape and volume_spacing.
    Returns:
        Projection matrices with shape (num_projections, 3, 4) as np.array.
    """

    #TODO: not working currently!
    # conrad docu:
    # https://www5.cs.fau.de/conrad/doc/edu/stanford/rsl/conrad/geometry/Projection.html

    projection_matrices = np.zeros((geometry.number_of_projections, 3, 4))

    # configure intrinsic camera parameters
    intrinsic_params_mat = np.eye(3, 3)
    for i in range(2):
        intrinsic_params_mat[i, i] = geometry.detector_spacing[1-i] * geometry.source_detector_distance
        intrinsic_params_mat[i, 2] = geometry.detector_origin[1-i]

    # configure projection
    projection = np.zeros((3, 4))
    for i in range(3):
        projection[i, i] = 1.0

    # configure extrinisc pararams and create projection_matrices
    current_angle = 0.0
    angular_increment = geometry.angular_range/geometry.number_of_projections
    for p in range(geometry.number_of_projections):
        # extrinsic rotation matrix about z-axis + translation x,y with source_isocenter_dist
        extrinsic_params_mat = np.eye(4, 4)
        extrinsic_params_mat[0, 0] = np.cos(current_angle)
        extrinsic_params_mat[1, 1] = extrinsic_params_mat[0, 0]
        extrinsic_params_mat[0, 1] = -np.sin(current_angle)
        extrinsic_params_mat[1, 0] = extrinsic_params_mat[0, 1]
        extrinsic_params_mat[0, 3] = geometry.source_isocenter_distance
        extrinsic_params_mat[1, 3] = geometry.source_isocenter_distance

        # calculate projection matrix
        projection_matrices[p] = np.dot(np.dot(intrinsic_params_mat, projection), extrinsic_params_mat)

        # next angle
        current_angle += angular_increment

    return projection_matrices

from deep_ct_reconstruction.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D

# TEST CODE! to be removed later on
if __name__ == '__main__':
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 100
    volume_shape = [volume_size, volume_size, volume_size]
    volume_spacing = [0.5, 0.5, 0.5]

    # Detector Parameters:
    detector_shape = [2*volume_size, 2*volume_size]
    detector_spacing = [0.5, 0.5]

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2 * np.pi

    source_detector_distance = 200
    source_isocenter_distance = 200

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.set_projection_matrices(circular_trajectory_3d_pyconrad(geometry))

    circular_trajectory_3d(geometry)