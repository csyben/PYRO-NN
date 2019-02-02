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


def circular_trajectory_3d_pyconrad(geometry):
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

    circ_traj.setPixelDimensionX(np.float64(geometry.detector_spacing[1]))
    circ_traj.setPixelDimensionY(np.float64(geometry.detector_spacing[0]))
    circ_traj.setDetectorHeight(int(geometry.detector_shape[0]))
    circ_traj.setDetectorWidth(int(geometry.detector_shape[1]))

    circ_traj.setOriginInPixelsX(np.float64(geometry.volume_origin[2]))
    circ_traj.setOriginInPixelsY(np.float64(geometry.volume_origin[1]))
    circ_traj.setOriginInPixelsZ(np.float64(geometry.volume_origin[0]))
    circ_traj.setReconDimensions(np.flip(geometry.volume_shape).tolist())
    circ_traj.setReconVoxelSizes(np.flip(geometry.volume_spacing).tolist())

    DETECTORMOTION_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 1)
    ROTATIONAXIS_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 3)

    average_angular_increment = np.degrees(geometry.angular_range/geometry.number_of_projections)
    detector_offset_u = 0
    detector_offset_v = 0
    rotationAxis = _.SimpleVector.from_list([0, 0, 1])
    center = _.PointND.from_list([0, 0, 0])

    circ_traj.setTrajectory(geometry.number_of_projections, geometry.source_isocenter_distance, average_angular_increment,
                            detector_offset_u, detector_offset_v, DETECTORMOTION_MINUS, ROTATIONAXIS_MINUS, rotationAxis, center, 0)

    for proj in range(0, geometry.number_of_projections):
        _projection_matrix[proj] = circ_traj.getProjectionMatrix(proj).computeP().as_numpy()

    return _projection_matrix


def circular_trajectory_3d(geometry):
    """
        Generates the projection matrices defining a circular trajectory around the z-axis
        for use with the 3d projection layers.
        Adapted from CONRAD Source code https://github.com/akmaier/CONRAD.
    Args:
        geometry: 3d Geometry class including angular_range, number_of_projections, source_detector_distance,
        detector_shape, detector_spacing, volume_origin, volume_shape and volume_spacing.
    Returns:
        Projection matrices with shape (num_projections, 3, 4) as np.array.
    """

    # init empty
    projection_matrices = np.zeros((geometry.number_of_projections, 3, 4))

    # axes for later use
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    # defining u and v directions by: main coord axes
    u_dir =  y_axis
    v_dir = -x_axis

    # configure intrinsic camera parameters
    intrinsic_params_mat = np.eye(3, 3)
    for i in range(2):
        intrinsic_params_mat[i, i] = geometry.source_detector_distance / geometry.detector_spacing[1-i]

    # calc and set detector origin
    intrinsic_params_mat[0:2, 2] = (geometry.detector_shape * 0.5)[::-1]


    # configure extrinisc pararams and create projection_matrices
    current_angle = 0.0
    angular_increment = geometry.angular_range/geometry.number_of_projections
    for p in range(geometry.number_of_projections):

        # calculate extrinsic params
        extrinsic_params_mat = np.eye(4, 4)

        # rotation of axes from world system to plane of rotation system
        R_to_plane = np.eye(4, 4)
        R_to_plane[0:3, 0:3] = np.array([z_axis, np.cross(z_axis, x_axis), -x_axis])

        # rotation for u and v direction
        axis_align_R = np.eye(4, 4)
        axis_align_R[0:3, 0] = u_dir
        axis_align_R[0:3, 1] = v_dir
        axis_align_R[0:3, 2] = np.cross(u_dir, v_dir)
        axis_align_R = axis_align_R.T

        # rotation about x axis
        R_x_axis = np.eye(4, 4)
        R_x_axis[0:3, 0:3] = np.array([1,                      0,                      0,
                                       0,  np.cos(-current_angle), -np.sin(-current_angle),
                                       0,  np.sin(-current_angle),  np.cos(-current_angle)]).reshape((3, 3))

        # translation of camera
        translation = np.eye(4, 4)
        translation[0:4, 3] = np.array([0, 0, geometry.source_isocenter_distance, 1])

        # combine the above into 4x4 extrinsic params matrix
        extrinsic_params_mat = np.dot(np.dot(np.dot(translation, axis_align_R), R_x_axis), R_to_plane)
        extrinsic_params_mat = extrinsic_params_mat / extrinsic_params_mat[3, 3]

        # calculate projection matrix
        projection_matrices[p][0:3, 0:3] = np.dot(intrinsic_params_mat, extrinsic_params_mat[0:3, 0:3])
        projection_matrices[p][0:3, 3] = np.dot(intrinsic_params_mat, extrinsic_params_mat[0:3, 3])

        # next angle
        current_angle += angular_increment

    return projection_matrices

from deep_ct_reconstruction.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D

# TEST CODE! to be removed later on
if __name__ == '__main__':
    # ------------------ Declare Parameters ------------------

    while(True):
        # Volume Parameters:
        volume_size = int(np.random.uniform(10, 1000))
        volume_shape = [1*volume_size, 2*volume_size, 3*volume_size]
        volume_spacing = [0.5, 0.5, 0.5]
        volume_spacing = np.random.uniform(0.1, 1, 3)

        # Detector Parameters:
        detector_shape = [4*volume_size, 5*volume_size]
        detector_spacing = [0.5, 0.5]
        detector_spacing = np.random.uniform(0.1, 1, 2)

        # Trajectory Parameters:
        number_of_projections = int(np.random.uniform(10, 1000))
        angular_range = 2 * np.pi

        source_detector_distance = int(np.random.uniform(10, 1000))
        source_isocenter_distance = int(np.random.uniform(10, 1000))

        # create Geometry class
        geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)

        # test vs pyconrad
        conrad_projection_matrices = circular_trajectory_3d_pyconrad(geometry)
        matrices = circular_trajectory_3d(geometry)

        overall_diff = conrad_projection_matrices-matrices
        overall_diff = np.sum(overall_diff)
        print(overall_diff)