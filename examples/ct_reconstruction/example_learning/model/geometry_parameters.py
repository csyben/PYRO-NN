import numpy as np
from deep_ct_reconstruction.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from deep_ct_reconstruction.ct_reconstruction.helpers.trajectories          import circular_trajectory


"""
    This file defines the Geometry parameters used by the hole model. 
    A GeometryParallel2D instance is provided to be used by everyone that needs it.
"""

# Declare Parameters
volume_shape          = [200, 200]
volume_spacing        = [0.5, 0.5]
detector_shape        = 300
detector_spacing      = 0.5
number_of_projections = 100
angular_range         = np.pi

# Create Geometry class instance
GEOMETRY = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
GEOMETRY.set_ray_vectors(circular_trajectory.circular_trajectory_2d(GEOMETRY))
