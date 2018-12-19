from tensorflow.python.framework import ops
import lme_custom_ops


# cone_projection3d
def cone_projection3d(volume,
                      volume_shape,
                      projection_shape,
                      volume_origin,
                      volume_spacing,
                      projection_matrices,
                      hardware_interp,
                      step_size):
    return lme_custom_ops.cone_projection3d(volume,
                                            volume_shape,
                                            projection_shape,
                                            volume_origin,
                                            volume_spacing,
                                            projection_matrices,
                                            hardware_interp,
                                            step_size)

def cone_projection3d(volume, geometry, hardware_interp, step_size):
    return lme_custom_ops.cone_projection3d(volume,
                                            geometry.volume_shape,
                                            geometry.projection_shape,
                                            geometry.tensor_proto_volume_origin,
                                            geometry.tensor_proto_volume_spacing,
                                            geometry.tensor_proto_projection_matrices,
                                            hardware_interp,
                                            step_size)

def cone_projection3d(volume, geometry):
    return lme_custom_ops.cone_projection3d(volume, **geometry.get_cone_projection3d_params_dict())

@ops.RegisterGradient( "ConeProjection3D" )
def _project_grad( op, grad ):
    reco = lme_custom_ops.cone_backprojection3d(
            sinogram                    = grad,
            sinogram_shape              = op.get_attr("projection_shape"),
            volume_shape                = op.get_attr("volume_shape"),
            volume_origin               = op.get_attr("volume_origin"),
            volume_spacing              = op.get_attr("volume_spacing"),
            projection_matrices         = op.get_attr("projection_matrices"),
            hardware_interp             = op.get_attr("hardware_interp"),
        )
    return [ reco ]