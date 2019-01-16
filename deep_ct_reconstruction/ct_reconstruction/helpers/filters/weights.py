import numpy as np


# code adapted from https://github.com/ma0ho/Deep-Learning-Cone-Beam-CT

# 3d cosine weights
def cosine_weights_3d(geometry):
    cu = geometry.detector_shape[1]/2 * geometry.detector_spacing[1]
    cv = geometry.detector_shape[0]/2 * geometry.detector_spacing[0]
    sd2 = geometry.source_detector_distance**2

    w = np.zeros((geometry.detector_shape[1], geometry.detector_shape[0]), dtype=np.float32)

    for v in range(0, geometry.detector_shape[0]):
        dv = ((v + 0.5) * geometry.detector_shape[0] - cv)**2
        for u in range(0, geometry.detector_shape[1]):
            du = ((u + 0.5) * geometry.detector_shape[1] - cu)**2
            w[v, u] = geometry.source_detector_distance / np.sqrt(sd2 + dv + dv)

    return w


#