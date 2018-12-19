from tensorflow.python.framework import ops
import lme_custom_ops


# parallel_backprojection2d
def parallel_backprojection2d(sinogram, geometry):
    return lme_custom_ops.parallel_backprojection2d(sinogram, **geometry.get_parallel_backprojection2d_params_dict())

'''
    Compute the gradient of the backprojector op by invoking the forward projector.
'''
@ops.RegisterGradient( "ParallelBackprojection2D" )
def _backproject_grad( op, grad ):
    proj = lme_custom_ops.parallel_projection2d(
            volume              = grad,
            volume_shape        = op.get_attr("volume_shape"),
            projection_shape    = op.get_attr("sinogram_shape"),
            volume_origin       = op.get_attr("volume_origin"),
            detector_origin     = op.get_attr("detector_origin"),
            volume_spacing      = op.get_attr("volume_spacing"),
            detector_spacing    = op.get_attr("detector_spacing"),
            ray_vectors         = op.get_attr("ray_vectors"),
        )
    return [ proj ]


# fan_backprojection2d
def fan_backprojection2d(sinogram, geometry):
    return lme_custom_ops.fan_backprojection2d(sinogram, **geometry.get_fan_backprojection2d_params_dict())

'''
    Compute the gradient of the backprojector op by invoking the forward projector.
'''
@ops.RegisterGradient( "FanBackprojection2D" )
def _backproject_grad( op, grad ):
    proj = lme_custom_ops.fan_projection2d(
            volume                      = grad,
            volume_shape                = op.get_attr("volume_shape"),
            projection_shape            = op.get_attr("sinogram_shape"),
            volume_origin               = op.get_attr("volume_origin"),
            detector_origin             = op.get_attr("detector_origin"),
            volume_spacing              = op.get_attr("volume_spacing"),
            detector_spacing            = op.get_attr("detector_spacing"),
            source_2_iso_distance       = op.get_attr("source_2_iso_distance"),
            source_2_detector_distance  = op.get_attr("source_2_detector_distance"),
            central_ray_vectors         = op.get_attr("central_ray_vectors"),
        )
    return [ proj ]
