from tensorflow.python.framework import ops
import pyronn_layers


# parallel_projection2d
def parallel_projection2d(volume, geometry):
    """
    Wrapper function for making the layer call.
    Args:
        volume:     Input volume to project.
        geometry:   Corresponding GeometryParallel2D Object defining parameters.
    Returns:
            Initialized lme_custom_ops.parallel_projection2d layer.
    """
    return pyronn_layers.parallel_projection2d(volume,
                                                volume_shape=geometry.volume_shape,
                                                projection_shape=geometry.sinogram_shape,
                                                volume_origin=geometry.tensor_proto_volume_origin,
                                                detector_origin=geometry.tensor_proto_detector_origin,
                                                volume_spacing=geometry.tensor_proto_volume_spacing,
                                                detector_spacing=geometry.tensor_proto_detector_spacing,
                                                ray_vectors=geometry.tensor_proto_ray_vectors)


@ops.RegisterGradient("ParallelProjection2D")
def _project_grad(op, grad):
    '''
        Compute the gradient of the projection op by invoking the backprojector.
    '''
    reco = pyronn_layers.parallel_backprojection2d(
        sinogram=grad,
        sinogram_shape=op.get_attr("projection_shape"),
        volume_shape=op.get_attr("volume_shape"),
        volume_origin=op.get_attr("volume_origin"),
        detector_origin=op.get_attr("detector_origin"),
        volume_spacing=op.get_attr("volume_spacing"),
        detector_spacing=op.get_attr("detector_spacing"),
        ray_vectors=op.get_attr("ray_vectors"),
    )
    return [reco]


# fan_projection2d
def fan_projection2d(volume, geometry):
    """
    Wrapper function for making the layer call.
    Args:
        volume:     Input volume to project.
        geometry:   Corresponding GeometryFan2D Object defining parameters.
    Returns:
            Initialized lme_custom_ops.fan_projection2d layer.
    """
    return pyronn_layers.fan_projection2d(volume,
                                           volume_shape=geometry.volume_shape,
                                           projection_shape=geometry.sinogram_shape,
                                           volume_origin=geometry.tensor_proto_volume_origin,
                                           detector_origin=geometry.tensor_proto_detector_origin,
                                           volume_spacing=geometry.tensor_proto_volume_spacing,
                                           detector_spacing=geometry.tensor_proto_detector_spacing,
                                           source_2_isocenter_distance=geometry.source_isocenter_distance,
                                           source_2_detector_distance=geometry.source_detector_distance,
                                           central_ray_vectors=geometry.tensor_proto_central_ray_vectors)


@ops.RegisterGradient("FanProjection2D")
def _project_grad(op, grad):
    '''
        Compute the gradient of the projection op by invoking the backprojector.
    '''
    reco = pyronn_layers.fan_backprojection2d(
        sinogram=grad,
        sinogram_shape=op.get_attr("projection_shape"),
        volume_shape=op.get_attr("volume_shape"),
        volume_origin=op.get_attr("volume_origin"),
        detector_origin=op.get_attr("detector_origin"),
        volume_spacing=op.get_attr("volume_spacing"),
        detector_spacing=op.get_attr("detector_spacing"),
        source_2_iso_distance=op.get_attr("source_2_iso_distance"),
        source_2_detector_distance=op.get_attr("source_2_detector_distance"),
        central_ray_vectors=op.get_attr("central_ray_vectors"),
    )
    return [reco]
