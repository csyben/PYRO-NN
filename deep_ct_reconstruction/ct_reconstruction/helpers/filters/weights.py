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
            w[v, u] = geometry.source_detector_distance / np.sqrt(sd2 + dv + du)

    return w

# parker weights
def parker_weights_2d(geometry):

    def parker_weights_fct(beta, gamma, fan_angle):
        if 0 <= beta and beta < 2 * (fan_angle + gamma):
            return np.sin((np.pi / 4.0) * beta / (fan_angle + gamma)) ** 2
        elif np.pi + 2 * gamma <= beta and beta <= np.pi + 2 * fan_angle:
            return np.sin((np.pi / 4.0) * (2 * fan_angle + np.pi - beta) / (fan_angle - gamma)) ** 2
        elif beta >= np.pi + 2 * fan_angle:
            return 0.0
        return 1.0

    weights = np.zeros(geometry.sinogram_shape)

    fan_angle = geometry.fan_angle

    gamma_range = np.linspace(-fan_angle, fan_angle, geometry.detector_shape[1], endpoint=False)
    beta_range = np.linspace(0, geometry.angular_range, geometry.number_of_projections, endpoint=False)

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = parker_weights_fct(beta_range[j], gamma_range[i], fan_angle)

    return weights



# riess weights
def riess_weights_2d(geometry):

    over = 2 * geometry.fan_angle

    def w1(b, a):
        x = np.pi + over - b
        y = over - 2 * a
        z = np.pi / 2 * (x / y)
        return np.pow(np.sin(z), 2)

    def w2(b, a):
        x = b
        y = over + 2 * a
        z = np.pi / 2 * (x / y)
        return np.pow(np.sin(z), 2)

    def riess_weights_fct(beta, gamma, fan_angle):
        if np.pi + 2 * fan_angle <= beta and beta <= np.pi + over:
            return w1(beta, gamma)
        elif np.pi + 2 * over - 2 * fan_angle <= beta and beta <= np.pi + over:
            return 2 - w1(beta, gamma)
        elif 0 <= beta and beta <= 2 * fan_angle + over:
            return w2(beta, gamma)
        elif 0 <= beta and beta <= -over - 2 * fan_angle:
            return 2 - w2(beta, gamma)
        else:
            return 1

    weights = np.zeros(geometry.sinogram_shape)

    fan_angle = geometry.fan_angle

    gamma_range = np.linspace(-fan_angle, fan_angle, geometry.detector_shape[1], endpoint=False)
    beta_range = np.linspace(0, geometry.angular_range, geometry.number_of_projections, endpoint=False)

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = riess_weights_fct(beta_range[j], gamma_range[i], fan_angle)

    return weights

