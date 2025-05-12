import numpy as np
# import trimesh
import matplotlib.pyplot as plt
from pyronn.ct_reconstruction.helpers.misc.general_utils import fibonacci_sphere, rotation_matrix_from_points

def arbitrary_projection_matrix(headers,voxel_size = [0.45,0.45], swap_detector_axis=False, **kwargs):
    #Source: Auto-calibration of cone beam geometries from arbitrary rotating markers using a vector geometry formulation of projection matrices by Graetz, Jonas
    number_of_projections = len(headers)
    # init empty
    projection_matrices = np.zeros((number_of_projections, 3, 4))

    detector_shape = np.array(
            [headers[0].number_vertical_pixels, headers[0].number_horizontal_pixels])
    
    # Shift into left upper corner of the detector
    detector_left_corner_trans = np.eye(3) 
    detector_left_corner_trans[0, 2] = + (float(headers[0].number_vertical_pixels) - 1.) / 2.
    detector_left_corner_trans[1, 2] = + (float( headers[0].number_horizontal_pixels) - 1.) / 2.
    detector_left_corner_trans[0, 0] *= 1
    detector_left_corner_trans[1, 1] *= -1
    detector_left_corner_trans[2, 2] = 1.
    traj_type = 'circ' if np.array_equal(np.array(headers[0].agv_source_position),np.array([0,0,0])) else 'free'
    print(traj_type)
    #Initial stuff for circular trajectory:
    if traj_type == 'circ':
        init_source_position = [0, 0, headers[0].focus_object_distance_in_mm]
        init_detector_position = [0, 0, headers[0].focus_object_distance_in_mm - headers[0].focus_detector_distance_in_mm]
        init_detector_line_direction = [0,1,0]
        init_detector_column_direction = [1,0,0]
        angular_range = headers[0].scan_range_in_rad
        if angular_range == 0:
            angular_range = 2 * np.pi
        current_angle = 0 
        angular_increment = angular_range/number_of_projections

    for p, header in enumerate(headers): 
        if traj_type == 'free':
            det_h = np.array(header.agv_detector_line_direction)
            det_v = -1* np.array(header.agv_detector_col_direction)
            source_center_in_voxel = (np.array(header.agv_source_position)/1000)/voxel_size[0] # in mm
            detector_center_in_voxel  = (np.array(header.agv_detector_center_position)/1000)/voxel_size[0] # in mm
        else:
            # rotation about x axis => Column direction of the detector
            R_x_axis = np.eye(3, 3)
            R_x_axis = np.array([1, 0, 0,
                                           0, np.cos(-current_angle), -np.sin(-current_angle),
                                           0, np.sin(-current_angle), np.cos(-current_angle)]).reshape((3, 3))
            source_center_in_voxel = np.dot(R_x_axis,init_source_position)/voxel_size[0]
            detector_center_in_voxel = np.dot(R_x_axis,init_detector_position)/voxel_size[0]
            det_h = np.dot(R_x_axis,init_detector_line_direction)
            det_v = np.dot(R_x_axis,init_detector_column_direction)
            current_angle += angular_increment

        #[H|V|d-s]
        h_v_sdd = np.column_stack((det_h, det_v, (detector_center_in_voxel - source_center_in_voxel) ))
        h_v_sdd_invers = np.linalg.inv(h_v_sdd)
        # [H|V|d-s]^-1 * -s
        back_part = h_v_sdd_invers @ (-source_center_in_voxel)
        proj_matrix = np.column_stack((h_v_sdd_invers,back_part))
        projection_matrices[p] =  detector_left_corner_trans @ proj_matrix
        
        # post processing to get the same oriented outputvolume like ezrt commandline reco: => tested, no changes needed to get the same orientation as Firefly ART
        # flip Z-Axis: Z = -Z
        if swap_detector_axis:
            projection_matrices[p][0:3, 2] = projection_matrices[p][0:3, 2] * -1.0

        # change orientation of current matrix from XYZ to YXZ: exchange the first two columns
        # projection_matrices[p][0:3, 0:2] = np.flip(projection_matrices[p][0:3, 0:2], axis=1)
        # change orientation of current matrix from YXZ to YZX: exchange the last two columns
        # projection_matrices[p][0:3, 1:3] = np.flip(projection_matrices[p][0:3, 1:3], axis=1)
    return projection_matrices


def fibonacci_sphere_projecton_matrix(number_of_projections, source_detector_distance,
                                      detector_spacing, source_isocenter_distance, detector_origin,
                                      swap_axis=False, *args, **kwargs):
    # init empty
    # assert len(pts) == number_of_projections
    pts = fibonacci_sphere(number_of_projections)
    projection_matrices = np.zeros((len(pts), 3, 4))
    x_axis = np.array([1., 0., 0.])
    y_axis = np.array([0., 1., 0.])
    z_axis = np.array([0., 0., 1.])

    u_dir = y_axis
    if swap_axis:
        v_dir = x_axis
    else:
        v_dir = -x_axis

    intrinsic_params_mat = np.eye(3, 3)
    for i in range(2):
        intrinsic_params_mat[i, i] = source_detector_distance / detector_spacing[1 - i]

    # calc and set detector origin
    # we need t_x and t_y, and follow the [z,y,x] convention

    intrinsic_params_mat[0, 2] = detector_origin[-1] / detector_spacing[-1] * -1
    intrinsic_params_mat[1, 2] = detector_origin[-2] / detector_spacing[-2] * -1

    for p in range(len(pts)):
        extrinsic_params_mat = np.eye(4, 4)

        R_to_plane = np.eye(4, 4)
        R_to_plane[0:3, 0:3] = np.array([z_axis, np.cross(z_axis, x_axis), -x_axis])


        axis_align_R = np.eye(4, 4)
        axis_align_R[0:3, 0] = u_dir
        axis_align_R[0:3, 1] = v_dir
        axis_align_R[0:3, 2] = np.cross(u_dir, v_dir)
        axis_align_R = axis_align_R.T

        translation = np.eye(4, 4)
        translation[0:4, 3] = np.array([0, 0, source_isocenter_distance, 1])

        R_to_pts = np.eye(4, 4)
        R_to_pts[0:3, 0:3] = rotation_matrix_from_points(pts[p],
                                                         np.array([0, 0, source_isocenter_distance])
                                                         )

        extrinsic_params_mat = np.dot(np.dot(np.dot(translation, axis_align_R), R_to_pts), R_to_plane)
        extrinsic_params_mat = extrinsic_params_mat / extrinsic_params_mat[3, 3]

        projection_matrices[p][0:3, 0:3] = np.dot(intrinsic_params_mat, extrinsic_params_mat[0:3, 0:3])
        projection_matrices[p][0:3, 3] = np.dot(intrinsic_params_mat, extrinsic_params_mat[0:3, 3])

    return projection_matrices
