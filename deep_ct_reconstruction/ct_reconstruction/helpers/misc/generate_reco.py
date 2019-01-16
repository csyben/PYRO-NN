import lme_custom_ops
import tensorflow as tf


def generate_reco(sinogram, layer, geometry):
    with tf.Session() as sess:
        result = layer(sinogram, geometry)
        sinogram = result.eval()
    return sinogram

def generate_reco_parallel_2d(phantom, geometry):
    result = lme_custom_ops.parallel_backprojection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram

def generate_reco_fan_2d(phantom, geometry):
    result = lme_custom_ops.fan_backprojection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram

def generate_reco_cone_3d(phantom, geometry):
    result = lme_custom_ops.cone_backprojection3d(phantom, geometry)
    sinogram = result.eval()
    return sinogram

