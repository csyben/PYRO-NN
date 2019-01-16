import lme_custom_ops


def generate_sinogram(phantom, layer, geometry):
    result = layer(phantom, geometry)
    sinogram = result.eval()
    return sinogram

def generate_sinogram_parallel_2d(phantom, geometry):
    result = lme_custom_ops.parallel_projection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram

def generate_sinogram_fan_2d(phantom, geometry):
    result = lme_custom_ops.fan_projection2d(phantom, geometry)
    sinogram = result.eval()
    return sinogram

def generate_sinogram_cone_3d(phantom, geometry):
    result = lme_custom_ops.cone_projection3d(phantom, geometry)
    sinogram = result.eval()
    return sinogram