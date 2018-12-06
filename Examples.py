import numpy as np
import tensorflow as tf
import Geometry
import lme_custom_ops
import pyconrad as pyc # TODO: get independent of pyconrad
pyc.setup_pyconrad()


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
    geometry = Geometry.Geometry_2d(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)

    # Get Phantom
    conrad_phantom_class = pyc.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
    phantom = conrad_phantom_class.SheppLogan(volume_size, False).as_numpy()
    pyc.imshow(phantom, 'phantom')


    # ------------------ Call Layers ------------------
    with tf.Session() as sess:
        result = lme_custom_ops.parallel_projection2d(phantom, *Geometry.get_parallel_projection2d_params(geometry))
        sinogram = result.eval()
        pyc.imshow(sinogram, 'sinogram')

        result_back_proj = lme_custom_ops.parallel_backprojection2d(sinogram, *Geometry.get_parallel_backprojection2d_params(geometry))
        reco = result_back_proj.eval()
        pyc.imshow(reco, 'reco')


def example_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 300
    volume_shape = [100, 200, 300]
    volume_spacing = [0.5, 0.5, 0.5]

    # Detector Parameters:
    detector_shape = [2*volume_size, 2*volume_size]
    detector_spacing = [0.5, 0.5]

    # Trajectory Parameters:
    number_of_projections = 100
    angular_range = np.pi

    source_detector_distance = 500
    source_isocenter_distance = 500

    # create Geometry class
    geometry = Geometry.Geometry_3d(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)

    # Get Phantom 3d
    _ = pyc.ClassGetter('edu.stanford.rsl.conrad.phantom')
    phantom = _.NumericalSheppLogan3D(*np.flip(volume_shape).tolist()).getNumericalSheppLoganPhantom().as_numpy()
    pyc.imshow(phantom, 'phantom')


    # ------------------ Call Layers ------------------
    with tf.Session() as sess:
        result = lme_custom_ops.cone_projection3d( phantom, geometry.volume_shape, geometry.sinogram_shape, geometry.tensor_proto_volume_origin, geometry.tensor_proto_volume_spacing, geometry.tensor_proto_projection_matrices, False, 0.5)
        sinogram = result.eval()
        pyc.imshow(sinogram, 'sinogram')

        #TODO: reco helper filters

        result_back_proj = lme_custom_ops.cone_backprojection3d(sinogram, geometry.sinogram_shape, geometry.volume_shape, geometry.tensor_proto_volume_origin, geometry.tensor_proto_volume_spacing, geometry.tensor_proto_projection_matrices, False)
        reco = result_back_proj.eval()
        pyc.imshow(reco, 'reco')


if __name__ == '__main__':
    example_2d()
    example_3d()