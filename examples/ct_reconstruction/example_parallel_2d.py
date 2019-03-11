import numpy as np
import tensorflow as tf
import lme_custom_ops
import pyconrad as pyc # TODO: get independent of pyconrad
pyc.setup_pyconrad()

# TODO: better imports
from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory


def example_parallel_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 512
    volume_shape = [volume_size, volume_size]
    volume_spacing = [0.5, 0.5]

    # Detector Parameters:
    detector_shape = 625
    detector_spacing = 0.5

    # Trajectory Parameters:
    number_of_projections = 30
    angular_range = np.pi

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan(volume_shape)
    pyc.imshow(phantom, 'phantom')


    # ------------------ Call Layers ------------------
    with tf.Session() as sess:
        result = parallel_projection2d(phantom, geometry)
        sinogram = result.eval()

        sinogram = sinogram + np.random.normal(
            loc=np.mean(np.abs(sinogram)), scale=np.std(sinogram), size=sinogram.shape) * 0.02

        pyc.imshow(sinogram, 'sinogram')

        reco_filter = filters.ram_lak_2D(geometry)

        sino_freq = np.fft.fft(sinogram, axis=1)
        sino_filtered_freq = np.multiply(sino_freq,reco_filter)
        sinogram_filtered = np.fft.ifft(sino_filtered_freq, axis=1)

        result_back_proj = parallel_backprojection2d(sinogram_filtered, geometry)
        reco = result_back_proj.eval()
        pyc.imshow(reco, 'reco')


if __name__ == '__main__':
    example_parallel_2d()
