import numpy as np
import tensorflow as tf
import lme_custom_ops
import pyconrad as pyc # TODO: get independent of pyconrad
pyc.setup_pyconrad()

# TODO: better imports
from deep_ct_reconstruction.ct_reconstruction.layers.projection_2d import parallel_projection2d
from deep_ct_reconstruction.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from deep_ct_reconstruction.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from deep_ct_reconstruction.ct_reconstruction.helpers.phantoms import shepp_logan
from deep_ct_reconstruction.ct_reconstruction.helpers.trajectories import circular_trajectory


def example_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 200
    volume_shape = [volume_size, volume_size]
    volume_spacing = [0.5, 0.5]

    # Detector Parameters:
    detector_shape = 2*volume_size
    detector_spacing = 0.5

    # Trajectory Parameters:
    number_of_projections = 100
    angular_range = np.pi

    source_detector_distance = None
    source_isocenter_distance = None

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, circular_trajectory.circular_trajectory_2d)
    geometry.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan(volume_size)
    pyc.imshow(phantom, 'phantom')


    # ------------------ Call Layers ------------------
    with tf.Session() as sess:
        result = parallel_projection2d(phantom, geometry)
        sinogram = result.eval()
        pyc.imshow(sinogram, 'sinogram')

        result_back_proj = parallel_backprojection2d(sinogram, geometry)
        reco = result_back_proj.eval()
        pyc.imshow(reco, 'reco')


if __name__ == '__main__':
    example_2d()
