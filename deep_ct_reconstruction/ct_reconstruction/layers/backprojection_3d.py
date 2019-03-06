from tensorflow.python.framework import ops
import lme_custom_ops

_used_geometery = None


# cone_backprojection3d
def cone_backprojection3d(sinogram, geometry, hardware_interp=True):
    """
    Wrapper function for making the layer call.
    Args:
        volume:             Input volume to project.
        geometry:           Corresponding GeometryCone3D Object defining parameters.
        hardware_interp:    Controls if interpolation is done by GPU 
    Returns:
            Initialized lme_custom_ops.cone_backprojection3d layer.
    """
    _used_geometery = geometry
    return lme_custom_ops.cone_backprojection3d(sinogram,
                                                sinogram_shape=geometry.sinogram_shape,
                                                volume_shape=geometry.volume_shape,
                                                volume_origin=geometry.tensor_proto_volume_origin,
                                                volume_spacing=geometry.tensor_proto_volume_spacing,
                                                projection_multiplier=geometry.projection_multiplier,
                                                projection_matrices=geometry.tensor_proto_projection_matrices,
                                                hardware_interp=hardware_interp)


'''
    Compute the gradient of the backprojector op by invoking the forward projector.
'''


@ops.RegisterGradient("ConeBackprojection3D")
def _backproject_grad(op, grad):
    proj = lme_custom_ops.cone_projection3d(
        volume=grad,
        volume_shape=op.get_attr("volume_shape"),
        projection_shape=op.get_attr("sinogram_shape"),
        volume_origin=op.get_attr("volume_origin"),
        volume_spacing=op.get_attr("volume_spacing"),
        projection_matrices=op.get_attr("projection_matrices"),
        hardware_interp=op.get_attr("hardware_interp"),
        step_size=_used_geometery.step_size,
    )
    return [proj]
