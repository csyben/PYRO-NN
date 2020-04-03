import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
from pyronn.ct_reconstruction.layers.projection_2d import fan_projection2d
from pyronn.ct_reconstruction.layers.projection_3d import cone_projection3d
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import fan_backprojection2d
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory


def example_parallel_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 64
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = 100
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = 90
    angular_range = np.pi

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape).astype(np.float32)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)
    sino = parallel_projection2d(phantom,geometry)
    @tf.function
    def test_func_proj(x):
        return parallel_projection2d(x,geometry)

    @tf.function
    def test_func_reco(x):
        return parallel_backprojection2d(x,geometry)

    proj_theoretical, proj_numerical = tf.test.compute_gradient(test_func_proj, [sino])
    reco_theoretical, reco_numerical = tf.test.compute_gradient(test_func_reco, [sino])

def example_fan_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 64
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = 100
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = 90
    angular_range = np.pi

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryFan2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape).astype(np.float32)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)
    sino = fan_projection2d(phantom,geometry)
    @tf.function
    def test_func_proj(x):
        return fan_projection2d(x,geometry)

    @tf.function
    def test_func_reco(x):
        return fan_backprojection2d(x,geometry)

    proj_theoretical, proj_numerical = tf.test.compute_gradient(test_func_proj, [sino])
    reco_theoretical, reco_numerical = tf.test.compute_gradient(test_func_reco, [sino])

def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 8
    volume_shape = [volume_size, volume_size, volume_size]
    volume_spacing = [1, 1, 1]

    # Detector Parameters:
    detector_shape = [12, 12]
    detector_spacing = [1,1]

    # Trajectory Parameters:
    number_of_projections = 12
    angular_range = np.pi

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_3d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_3d(volume_shape).astype(np.float32)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)
    sino = cone_projection3d(phantom,geometry)
    @tf.function
    def test_func_proj(x):
        return cone_projection3d(x,geometry)

    @tf.function
    def test_func_reco(x):
        return cone_backprojection3d(x,geometry)

    proj_theoretical, proj_numerical = tf.test.compute_gradient(test_func_proj, [sino])
    reco_theoretical, reco_numerical = tf.test.compute_gradient(test_func_reco, [sino])

if __name__ == '__main__':
    example_parallel_2d()
    example_fan_2d()
    example_cone_3d()