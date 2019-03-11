import tensorflow as tf
from ...layers.projection_2d import parallel_projection2d
from ...layers.projection_2d import fan_projection2d
from ...layers.projection_3d import cone_projection3d


def generate_sinogram(phantom, layer, geometry):
    with tf.Session() as sess:
        result = layer(phantom, geometry)
        sinogram = result.eval()
    return sinogram


def generate_sinogram_parallel_2d(phantom, geometry):
    result = parallel_projection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram


def generate_sinogram_fan_2d(phantom, geometry):
    result = fan_projection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram


def generate_sinogram_cone_3d(phantom, geometry):
    result = cone_projection3d(phantom, geometry)
    sinogram = result.eval()
    return sinogram
