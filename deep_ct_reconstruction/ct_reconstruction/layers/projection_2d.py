from tensorflow.python.framework import ops
import lme_custom_ops


# parallel_projection2d
def parallel_projection2d(volume,
                          volume_shape,     
                          projection_shape, 
                          volume_origin,    
                          detector_origin,  
                          volume_spacing,   
                          detector_spacing, 
                          ray_vectors):
    return lme_custom_ops.parallel_projection2d(volume,
                                                volume_shape,     
                                                projection_shape, 
                                                volume_origin,    
                                                detector_origin,  
                                                volume_spacing,   
                                                detector_spacing, 
                                                ray_vectors)

def parallel_projection2d(volume, geometry):
    return lme_custom_ops.parallel_projection2d(volume, **geometry.get_parallel_projection2d_params_dict())

'''
    Compute the gradient of the projection op by invoking the backprojector.
'''
@ops.RegisterGradient( "ParallelProjection2D" )
def _project_grad( op, grad ):
    reco = lme_custom_ops.parallel_backprojection2d(
            sinogram                      = grad,
            sinogram_shape              = op.get_attr("projection_shape"),
            volume_shape                = op.get_attr("volume_shape"),
            volume_origin               = op.get_attr("volume_origin"),
            detector_origin             = op.get_attr("detector_origin"),
            volume_spacing              = op.get_attr("volume_spacing"),
            detector_spacing            = op.get_attr("detector_spacing"),
            ray_vectors                 = op.get_attr("ray_vectors"),
        )
    return [ reco ]


# fan_projection2d
def fan_projection2d(volume,
                     volume_shape,
                     projection_shape,
                     volume_origin,
                     detector_origin,
                     volume_spacing,
                     detector_spacing,
                     source_2_iso_distance,
                     source_2_detector_distance,
                     central_ray_vectors):
    return lme_custom_ops.fan_projection2d(volume,
                                           volume_shape,
                                           projection_shape,
                                           volume_origin,
                                           detector_origin,
                                           volume_spacing,
                                           detector_spacing,
                                           source_2_iso_distance,
                                           source_2_detector_distance,
                                           central_ray_vectors)

def fan_projection2d(volume, geometry):
    return lme_custom_ops.fan_projection2d(volume,
                                           geometry.volume_shape,
                                           geometry.projection_shape,
                                           geometry.tensor_proto_volume_origin,
                                           geometry.tensor_proto_detector_origin,
                                           geometry.tensor_proto_volume_spacing,
                                           geometry.tensor_proto_detector_spacing,
                                           geometry.source_2_iso_distance,
                                           geometry.source_2_detector_distance,
                                           geometry.tensor_proto_.central_ray_vectors)

'''
    Compute the gradient of the projection op by invoking the backprojector.
'''
@ops.RegisterGradient( "FanProjection2D" )
def _project_grad( op, grad ):
    reco = lme_custom_ops.fan_backprojection2d(
            sinogram                      = grad,
            sinogram_shape              = op.get_attr("projection_shape"),
            volume_shape                = op.get_attr("volume_shape"),
            volume_origin               = op.get_attr("volume_origin"),
            detector_origin             = op.get_attr("detector_origin"),
            volume_spacing              = op.get_attr("volume_spacing"),
            detector_spacing            = op.get_attr("detector_spacing"),
            source_2_iso_distance       = op.get_attr("source_2_iso_distance"),
            source_2_detector_distance  = op.get_attr("source_2_detector_distance"),
            central_ray_vectors         = op.get_attr("central_ray_vectors"),
        )
    return [ reco ]