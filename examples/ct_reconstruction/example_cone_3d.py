import numpy as np
import tensorflow as tf
import lme_custom_ops
import pyconrad as pyc # TODO: get independent of pyconrad
pyc.setup_pyconrad()

# TODO: better imports
from pyronn.ct_reconstruction.layers.projection_3d import cone_projection3d
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters.filters import ramp


def example_cone_3d():
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
    geometry.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geometry))

    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    pyc.imshow(phantom, 'phantom')


    # ------------------ Call Layers ------------------
    with tf.Session() as sess:
        result = cone_projection3d(phantom, geometry)
        sinogram = result.eval()
        pyc.imshow(sinogram, 'sinogram')

        # filtering
        filter = ramp(int(geometry.detector_shape[1]))
        sino_freq = np.fft.fft(sinogram, axis=2)
        filtered_sino_freq = np.zeros_like(sino_freq)
        for row in range(int(geometry.detector_shape[0])):
            for projection in range(geometry.number_of_projections):
                filtered_sino_freq[projection, row, :] = sino_freq[projection, row, :] * filter[:]

        filtered_sino = np.fft.ifft(filtered_sino_freq, axis=2)

        result_back_proj = cone_backprojection3d(filtered_sino, geometry)
        reco = result_back_proj.eval()
        pyc.imshow(reco, 'reco')


if __name__ == '__main__':
    example_cone_3d()
