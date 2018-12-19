from tensorflow.python.framework import ops
import lme_custom_ops


# cone_backprojection3d
def cone_backprojection3d(sinogram, geometry):
    return lme_custom_ops.cone_backprojection3d(sinogram, **geometry.get_cone_backprojection3d_params_dict())

'''
    Compute the gradient of the backprojector op by invoking the forward projector.
'''
@ops.RegisterGradient( "ConeBackprojection3D" )
def _backproject_grad( op, grad ):
    proj = lme_custom_ops.cone_projection3d(
            volume              = grad,
            volume_shape        = op.get_attr("volume_shape"),
            projection_shape    = op.get_attr("sinogram_shape"),
            volume_origin       = op.get_attr("volume_origin"),
            volume_spacing      = op.get_attr("volume_spacing"),
            projection_matrices = op.get_attr("projection_matrices"),
            hardware_interp     = op.get_attr("hardware_interp"),
            step_size           = op.get_attr("step_size"),
        )
    return [ proj ]