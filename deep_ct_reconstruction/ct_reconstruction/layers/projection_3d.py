from tensorflow.python.framework import ops
import lme_custom_ops

_used_geometery = None
# cone_projection3d
def cone_projection3d(volume, geometry, hardware_interp=True, step_size=1.0):
    """
    Wrapper function for making the layer call.
    Args:
        volume:             Input volume to project.
        geometry:           Corresponding GeometryCone3D Object defining parameters.
        hardware_interp:    Controls if interpolation is done by GPU.
        step_size:          step_size along ray direction in voxel.
    Returns:
            Initialized lme_custom_ops.cone_projection3d layer.
    """
    _used_geometery = geometry
    return lme_custom_ops.cone_projection3d(volume,
                                            volume_shape        = geometry.volume_shape,
                                            projection_shape    = geometry.sinogram_shape,
                                            volume_origin       = geometry.tensor_proto_volume_origin,
                                            volume_spacing      = geometry.tensor_proto_volume_spacing,
                                            projection_matrices = geometry.tensor_proto_projection_matrices,
                                            hardware_interp     = hardware_interp,
                                            step_size           = step_size)

'''
    Compute the gradient of the projection op by invoking the backprojector.
'''
@ops.RegisterGradient( "ConeProjection3D" )
def _project_grad( op, grad ):
    reco = lme_custom_ops.cone_backprojection3d(
            sinogram                = grad,
            sinogram_shape          = op.get_attr("projection_shape"),
            volume_shape            = op.get_attr("volume_shape"),
            volume_origin           = op.get_attr("volume_origin"),
            volume_spacing          = op.get_attr("volume_spacing"),
            projection_multiplier   = _used_geometery.projection_multipier,
            projection_matrices     = op.get_attr("projection_matrices"),
            hardware_interp         = op.get_attr("hardware_interp"),
        )
    return [ reco ]