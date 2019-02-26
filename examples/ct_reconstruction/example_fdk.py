import numpy as np
import tensorflow as tf
import math
import lme_custom_ops

# TODO: better imports
from deep_ct_reconstruction.ct_reconstruction.helpers.misc import generate_sinogram
from deep_ct_reconstruction.ct_reconstruction.layers.projection_3d import cone_projection3d
from deep_ct_reconstruction.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from deep_ct_reconstruction.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from deep_ct_reconstruction.ct_reconstruction.helpers.phantoms import shepp_logan
from deep_ct_reconstruction.ct_reconstruction.helpers.trajectories import circular_trajectory
from deep_ct_reconstruction.ct_reconstruction.helpers.filters.filters import ram_lak_3D
from deep_ct_reconstruction.ct_reconstruction.helpers.filters.filters import ramp_3D

import deep_ct_reconstruction.ct_reconstruction.helpers.filters.weights as ct_weights

import pyconrad as pyc
pyc.setup_pyconrad()
pyc.start_gui()

class nn_model:
    def __init__(self, geometry):
        self.geometry = geometry

        self.cosine_weight = tf.get_variable(name='cosine_weight', dtype=tf.float32,
                                             initializer=ct_weights.cosine_weights_3d(self.geometry), trainable=False)

        #TODO: Check primary angles, array should be based on proj mat
        primary_angles = np.arange(0, geometry.angular_range, 0.014067353771074298)[
                         0:geometry.number_of_projections]
        primary_angles_2 = np.linspace(0, geometry.angular_range, geometry.number_of_projections)

        self.redundancy_weight = tf.get_variable(name='redundancy_weight', dtype=tf.float32,
                                             initializer=ct_weights.init_parker_3D(self.geometry,primary_angles), trainable=False)

        self.filter = tf.get_variable(name='reco_filter', dtype=tf.float32, initializer=ram_lak_3D(self.geometry), trainable=False)



    def model(self, sinogram):
        self.sinogram_cos = tf.multiply(sinogram, self.cosine_weight)
        self.redundancy_weighted_sino = tf.multiply(self.sinogram_cos,self.redundancy_weight)

        self.weighted_sino_fft = tf.fft(tf.cast(self.redundancy_weighted_sino, dtype=tf.complex64))
        self.filtered_sinogram_fft = tf.multiply(self.weighted_sino_fft, tf.cast(self.filter,dtype=tf.complex64))
        self.filtered_sinogram = tf.real(tf.ifft(self.filtered_sinogram_fft))

        self.reconstruction = cone_backprojection3d(self.filtered_sinogram,self.geometry, hardware_interp=True)

        return self.reconstruction, self.redundancy_weighted_sino


def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size, volume_size]
    volume_spacing = [0.5, 0.5, 0.5]

    # Detector Parameters:
    detector_shape = [400 , 400]#[np.ceil(np.sqrt(2*np.power(volume_size,2))).astype(np.int32)+2,np.ceil(np.sqrt(2*np.power(volume_size,2))).astype(np.int32)+2]
    detector_spacing = [0.5, 0.5]

    # Trajectory Parameters:
    number_of_projections = 248
    angular_range = math.radians(200) #200 * np.pi / 180

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    #geometry.set_projection_matrices(test_trajectory())
    proj, angles = test_trajectory()
    proj2 =  circular_trajectory.circular_trajectory_3d(geometry)
    proj3 = circular_trajectory_3d_pyconrad(geometry)
    print('proj2')
    print(proj2[0])
    print('shape2')
    print(np.shape(proj2))
    geometry.set_projection_matrices(proj)
    #geometry.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geometry))
    #print(geometry.projection_matrices[0])
    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    pyc.imshow(phantom, 'phantom')

    sinogram = generate_sinogram.generate_sinogram(phantom,cone_projection3d,geometry)
    pyc.imshow(sinogram, 'sinogram')

    # ------------------ Call Layers ------------------
    with tf.Session() as sess:

        model = nn_model(geometry)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        reco_tf, redundancy_weighted_sino_tf = model.model(sinogram)
        reco, redundancy_weighted_sino = sess.run([reco_tf, redundancy_weighted_sino_tf])
    pyc.imshow(reco,'reco')
    pyc.imshow(redundancy_weighted_sino, 'redundancy_weighted_sino')
    a=5

def test_trajectory():
    _ = pyc.ClassGetter('edu.stanford.rsl.conrad.geometry.trajectories', 'edu.stanford.rsl.conrad.geometry')

    circ_traj = pyc.edu().stanford.rsl.conrad.geometry.trajectories.CircularTrajectory()

    circ_traj = _.CircularTrajectory()
    circ_traj.setSourceToDetectorDistance(1200)

    circ_traj.setPixelDimensionX(0.5)
    circ_traj.setPixelDimensionY(0.5)
    circ_traj.setDetectorHeight(400)
    circ_traj.setDetectorWidth(400)

    circ_traj.setOriginInPixelsX((256 - 1) / 2.0)
    circ_traj.setOriginInPixelsY((256 - 1) / 2.0)
    circ_traj.setOriginInPixelsZ((256 - 1) / 2.0)
    circ_traj.setReconDimensions([256, 256, 256])
    circ_traj.setReconVoxelSizes(np.array([0.5, 0.5, 0.5]))

    DETECTORMOTION_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 0)
    ROTATIONAXIS_MINUS = _.enumval_from_int('Projection$CameraAxisDirection', 2)

    average_angular_increment = 0.806
    detector_offset_u = 0
    detector_offset_v = 0
    rotationAxis = _.SimpleVector.from_list([0, 0, 1])
    center = _.PointND.from_list([0, 0, 0])

    number_of_projections = 248

    circ_traj.setTrajectory(number_of_projections , 750, average_angular_increment,
                            detector_offset_u, detector_offset_v, DETECTORMOTION_MINUS, ROTATIONAXIS_MINUS,
                            rotationAxis, center, 0)

    _projection_matrix = np.zeros((number_of_projections, 3, 4))
    primary_angles = np.array(circ_traj.getPrimaryAngles())
    for proj in range(0, number_of_projections):
        _projection_matrix[proj] = circ_traj.getProjectionMatrix(proj).computeP().as_numpy()

    print('TRAJ TEST')
    print(_projection_matrix[0])
    print('shape')
    print(np.shape(_projection_matrix))
    a = 5
    return _projection_matrix, primary_angles

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



if __name__ == '__main__':
    example_cone_3d()
