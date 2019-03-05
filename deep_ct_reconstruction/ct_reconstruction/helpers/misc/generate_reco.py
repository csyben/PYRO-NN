import tensorflow as tf
from ...layers.backprojection_2d import parallel_backprojection2d
from ...layers.backprojection_2d import fan_backprojection2d
from ...layers.backprojection_3d import cone_backprojection3d


def generate_reco(sinogram, layer, geometry):
    with tf.Session() as sess:
        result = layer(sinogram, geometry)
        sinogram = result.eval()
    return sinogram


def generate_reco_parallel_2d(phantom, geometry):
    result = parallel_backprojection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram


def generate_reco_fan_2d(phantom, geometry):
    result = fan_backprojection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram


def generate_reco_cone_3d(phantom, geometry):
    result = cone_backprojection3d(phantom, geometry)
    sinogram = result.eval()
    return sinogram
