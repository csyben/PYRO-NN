import numpy as np


def sphere(shape, pos, radius, value=1.0):
    """
        Creates a simple sphere primitive.
    Args:
        shape:      Shape (in [Z, Y, X])
        pos:        Center (in [Z, Y, X]) from upper left corner
        radius:     Radius
        value:      Value

    Returns:
        np.array filled with sphere
    """
    # create meshgrid of coords
    xx, yy, zz = np.mgrid[:shape[0], :shape[1], :shape[2]]

    # calc squared distance to pos
    circle = (xx - pos[2]) ** 2 + (yy - pos[1]) ** 2 + (zz - pos[0]) ** 2

    return (circle <= radius**2) * value


def ellipsoid(shape, pos, half_axes, value=1.0, euler_angles=(0.0, 0.0, 0.0)):
    """

    Args:
        shape:        Shape (in [Z, Y, X])
        pos:          Center (in [Z, Y, X]) from upper left corner
        half_axes:    Half axes of the ellipse (in [b, a, c])
        value:        Value
        euler_angles: The euler angles [phi, theta, psi].
                      Will define a Rotation Matrix using convention R = Rz(phi) * Ry(theta) * Rz(psi)
                      With: Rx(angle) = [[ 1,  0,  0 ],
                                         [ 0,  c, -s ],
                                         [ 0,  s,  c ]]
                            Ry(angle) = [[ c,  0,  s ],
                                         [ 0,  1,  0 ],
                                         [-s,  0,  c ]]
                            Rz(angle) = [[ c, -s,  0 ],
                                         [ s,  c,  0 ],
                                         [ 0,  0,  1 ]]
                            where s = sine(angle) and c = cosine(angle)

    Returns:
        np.array filled with ellipsoid
    """
    # create meshgrid of coords
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]

    # move to pos
    xc = (xx - pos[2])
    yc = (yy - pos[1])
    zc = (zz - pos[0])

    # build up euler rotation
    c = np.cos(euler_angles[0])
    s = np.sin(euler_angles[0])
    Rz_phi   = np.array([[ c, -s,  0 ],
                         [ s,  c,  0 ],
                         [ 0,  0,  1 ]])
    c = np.cos(euler_angles[1])
    s = np.sin(euler_angles[1])
    Ry_theta = np.array([[ c,  0,  s ],
                         [ 0,  1,  0 ],
                         [-s,  0,  c ]])
    c = np.cos(euler_angles[2])
    s = np.sin(euler_angles[2])
    Rz_psi   = np.array([[ c, -s,  0 ],
                         [ s,  c,  0 ],
                         [ 0,  0,  1 ]])

    # R = Rz(phi) * Ry(theta) * Rz(psi)
    R = np.dot(np.dot(Rz_phi, Ry_theta), Rz_psi).T

    xx = xc * R[0, 0] + yc * R[0, 1] + zc * R[0, 2]
    yy = xc * R[1, 0] + yc * R[1, 1] + zc * R[1, 2]
    zz = xc * R[2, 0] + yc * R[2, 1] + zc * R[2, 2]

    a = half_axes[2]
    b = half_axes[1]
    c = half_axes[0]

    # calc squared distance to pos
    ellipse_points = (xx ** 2) / (a**2) + (yy ** 2) / (b**2) + (zz ** 2) / (c**2)

    return (ellipse_points <= 1) * value


def cube(shape, pos, size, value=1.0):
    """
        Creates a simple cube primitive.
    Args:
        shape:      Shape (in [Z, Y, X])
        pos:        Pos (upper left corner) (in [Z, Y, X]) from upper left corner
        size:       Size  (in [Z, Y, X])
        value:      Value

    Returns:
        np.array filled with cube
    """
    # create array and populate it with value
    the_cube = np.zeros(shape)
    the_cube[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1], pos[2]:pos[2]+size[2]] = value

    return the_cube
