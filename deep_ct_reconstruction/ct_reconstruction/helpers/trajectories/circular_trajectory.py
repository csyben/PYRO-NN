import numpy as np
import pyconrad as pyc  # TODO: get independent of pyconrad


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


def circular_trajectory_3d(geometry): # TODO: still have to get independent of pyconrad
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
