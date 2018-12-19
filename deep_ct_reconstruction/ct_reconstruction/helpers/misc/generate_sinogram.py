import lme_custom_ops


def generate_sinogram(phantom, layer, geometry_params_dict):
    result = layer(phantom, geometry_params_dict)
    sinogram = result.eval()
    return sinogram

def generate_sinogram_parallel_2d(phantom, geometry):
    result = lme_custom_ops.parallel_projection2d(phantom, *geometry.get_parallel_projection2d_params_dict())
    sinogram = result.eval()
    return sinogram

def generate_sinogram_fan_2d(phantom, geometry):
    result = lme_custom_ops.fan_projection2d(phantom, *geometry.get_fan_projection2d_params_dict())
    sinogram = result.eval()
    return sinogram

def generate_sinogram_cone_3d(phantom, geometry):
    result = lme_custom_ops.cone_projection3d(phantom, *geometry.get_cone_projection3d_params_dict())
    sinogram = result.eval()
    return sinogram