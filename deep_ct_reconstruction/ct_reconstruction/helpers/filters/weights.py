import numpy as np
import math

eps = 10e-7
# code adapted from https://github.com/ma0ho/Deep-Learning-Cone-Beam-CT

# 3d cosine weights
def cosine_weights_3d(geometry):
    cu = (geometry.detector_shape[1]-1)/2 * geometry.detector_spacing[1]
    cv = (geometry.detector_shape[0]-1)/2 * geometry.detector_spacing[0]
    sd2 = geometry.source_detector_distance**2

    w = np.zeros((geometry.detector_shape[1], geometry.detector_shape[0]), dtype=np.float32)

    for v in range(0, geometry.detector_shape[0]):
        dv = ((v + 0.5) * geometry.detector_spacing[0] - cv)**2
        for u in range(0, geometry.detector_shape[1]):
            du = ((u + 0.5) * geometry.detector_spacing[1] - cu)**2
            w[v, u] = geometry.source_detector_distance / np.sqrt(sd2 + dv + du)

    return np.flip(w)

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

    weights = np.zeros(geometry.sinogram_shape.astype(np.int32))

    fan_angle = geometry.fan_angle

    gamma_range = np.linspace(-fan_angle, fan_angle, geometry.detector_shape[1], endpoint=False)
    beta_range = np.linspace(0, geometry.angular_range, geometry.number_of_projections, endpoint=False)

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = parker_weights_fct(beta_range[j], gamma_range[i], fan_angle)

    return weights

def init_parker_1D( geometry, beta, delta ):

    detector_width = geometry.detector_shape[-1].astype(np.int32)
    detector_spacing_width = geometry.detector_spacing[-1]

    w = np.ones( ( geometry.detector_shape[-1].astype(np.int32) ), dtype = np.float32 )

    for u in range( 0, detector_width ):
        # current fan angle
        alpha = math.atan( ( u+0.5 -(detector_width)/2.0 ) * detector_spacing_width / geometry.source_detector_distance )

        if beta >= 0 and beta < 2 * (delta+alpha):
            # begin of scan
            w[u] = math.pow( math.sin( math.pi/4 * ( beta / (delta+alpha) ) ), 2 )
        elif beta >= math.pi + 2*alpha and beta < math.pi + 2*delta:
            # end of scan
            w[u] = math.pow( math.sin( math.pi/4 * ( ( math.pi + 2*delta - beta ) / ( delta - alpha ) ) ), 2 )
        elif beta >= math.pi + 2*delta:
            # out of range
            w[u] = 0.0

    return w#np.flip(w)


def init_parker_3D( geometry, primary_angles_rad ):
    detector_width = geometry.detector_shape[np.alen(geometry.detector_shape) - 1].astype(np.int32)
    detector_spacing_width = geometry.detector_spacing[np.alen(geometry.detector_spacing) - 1]

    pa = primary_angles_rad

    # normalize angles to [0, 2*pi]
    pa -= pa[0]
    pa = np.where( pa < 0, pa + 2*math.pi, pa )

    # find rotation such that max(angles) is minimal
    tmp = np.reshape( pa, ( pa.size, 1 ) ) - pa
    tmp = np.where( tmp < 0, tmp + 2*math.pi, tmp )
    pa = tmp[:, np.argmin( np.max( tmp, 0 ) )]

    # delta = maximum fan_angle
    delta = math.atan( ( float((detector_width) * detector_spacing_width) / 2 ) / geometry.source_detector_distance )
    t_range = np.max(pa)
    max_range = math.pi + 2 * delta
    offset = (max_range - t_range) / 2

    factor = (t_range + t_range/geometry.number_of_projections) / math.pi

    f = lambda pi: init_parker_1D( geometry, pi, delta )


    # go over projections
    w = [
            np.reshape(
                f( pa[i] + offset) *factor,
                (1, 1, detector_width)
            )
            for i in range( 0 , pa.size)
    ]

    w = np.concatenate( w )

    return w


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

def testParker3d():
    from deep_ct_reconstruction.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
    from deep_ct_reconstruction.ct_reconstruction.helpers.trajectories import circular_trajectory
    import pyconrad as pyc
    pyc.setup_pyconrad()
    pyc.start_gui()

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size, volume_size]
    v_spacing = 0.5
    volume_spacing = [v_spacing,v_spacing,v_spacing]

    # Detector Parameters:
    detector_shape = [500 , 500]
    d_spacing = 0.5
    detector_spacing = [d_spacing,d_spacing]

    # Trajectory Parameters:
    number_of_projections = 248
    angular_range = math.radians(200) #200 * np.pi / 180

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geometry))

    primary_angles_2 = np.linspace(0, geometry.angular_range, geometry.number_of_projections)

    our_weights = np.squeeze(init_parker_3D(geometry, primary_angles_2)).T
    conrad_weights = parker_weights_conrad(geometry)
    diff = our_weights - conrad_weights

    pyc.imshow(our_weights, 'our_weights')
    pyc.imshow(conrad_weights, 'conrad_weights')
    pyc.imshow(diff, 'diff')


def parker_weights_conrad_3d(geometry):
    return np.array(np.expand_dims(parker_weights_conrad(geometry).T, axis=1), dtype=np.float32)


def parker_weights_conrad(geometry):

    focalLength = geometry.source_detector_distance
    maxT = geometry.detector_shape[1]*geometry.detector_spacing[1]
    deltaT = geometry.detector_spacing[1]
    maxBeta = geometry.angular_range
    deltaBeta = geometry.angular_range/geometry.number_of_projections

    # Initialize parameters
    maxBetaIndex = int(np.round(maxBeta / deltaBeta))
    maxTIndex = int(np.round(maxT / deltaT))
    gammaM =  np.arctan((maxT / 2.0)/ focalLength)
    beta = 0
    alpha = 0

    output_weights = np.zeros((maxTIndex, maxBetaIndex))

    # iterate over the detector elements
    for t in range(maxTIndex):
        # compute alpha of the current ray (detector element)
        alpha = np.arctan((t * deltaT - maxT / 2.0 + 0.5*deltaT) / focalLength)
        
        # iterate over the projection angles
        for b in range(maxBetaIndex):
            beta = b * deltaBeta
            
            # Shift weights such that they are centered (Important for maxBeta < pi + 2 * gammaM)
            beta += (np.pi+2*gammaM-maxBeta)/2.0
            
            # Adjust beta if out of range [0, 2*pi]
            if (beta < 0): 
                continue
            
            if (beta > np.pi *2.0): 
                continue
            
            # implement the conditions as described in Parker's paper
            if (beta <= 2 * (gammaM - alpha)): 
                tmp = beta * np.pi / 4.0 / (gammaM - alpha)
                val = np.sin(tmp)**2.0
                
                if (val is np.nan):
                    continue
                
                output_weights[t, b] = val

            elif (beta < np.pi - 2.0 * alpha):
                output_weights[t, b] = 1
            
            elif (beta <= (np.pi + 2.0 * gammaM) + 1e-12): 
                tmp = (np.pi / 4.0) * ( (np.pi + 2.0*gammaM - beta) / (gammaM + alpha) )
                val = np.sin(tmp)**2.0
                if (val is np.nan):
                    continue

                output_weights[t, b] = val
   
    # Correct for scaling due to varying angle
    output_weights = output_weights *  maxBeta / np.pi

    return output_weights


def testParker2d():
    from deep_ct_reconstruction.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D
    from deep_ct_reconstruction.ct_reconstruction.helpers.trajectories import circular_trajectory

    # Volume Parameters:
    volume_size = 512
    volume_shape = [volume_size, volume_size]
    volume_spacing = [0.5, 0.5]

    # Detector Parameters:
    detector_shape = 800
    detector_spacing = 0.5

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2 * np.pi

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryFan2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.set_central_ray_vectors(circular_trajectory.circular_trajectory_2d(geometry))


if __name__ == '__main__':
    testParker3d()
    testParker2d()
