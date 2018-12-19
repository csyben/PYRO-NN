import lme_custom_ops


def generate_reco(sinogram, layer, geometry_params_dict):
    result = layer(sinogram, geometry_params_dict)
    sinogram = result.eval()
    return sinogram

def generate_reco_parallel_2d(phantom, geometry):
    result = lme_custom_ops.parallel_backprojection2d(phantom, *geometry.get_parallel_projection2d_params_dict())
    sinogram = result.eval()
    return sinogram

def generate_reco_fan_2d(phantom, geometry):
    result = lme_custom_ops.fan_backprojection2d(phantom, *geometry.get_fan_projection2d_params_dict())
    sinogram = result.eval()
    return sinogram

def generate_reco_cone_3d(phantom, geometry):
    result = lme_custom_ops.cone_backprojection3d(phantom, *geometry.get_cone_projection3d_params_dict())
    sinogram = result.eval()
    return sinogram

