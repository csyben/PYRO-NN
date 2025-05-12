# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def shepp_logan(shape):
    """
        Creates the Shepp Logan Phantom.

    Args:
        shape: Shape (in [Y, X]) of Shepp Logan phantom to create.

    Returns:
        Shepp Logan of shape as np.array

    """
    # Definition of Shepp Logan Phantom
    # Ellipse	Center	         Major Axis    Minor Axis    Phi     Gray Level
    # a	        (0,0)	         0.69	       0.92	         0	     2
    # b	        (0,-0.0184)	     0.6624	       0.874	     0	     -0.98
    # c	        (0.22,0)	     0.11	       0.31	         -18°	 -0.02
    # d	        (-0.22,0)	     0.16	       0.41	         18°	 -0.02
    # e	        (0,0.35)	     0.21	       0.25	         0	     0.01
    # f	        (0,0.1)	         0.046	       0.046	     0	     0.01
    # g	        (0,-0.1)	     0.046	       0.046	     0	     0.01
    # h	        (-0.08,-0.605)	 0.046	       0.023	     0	     0.01
    # i	        (0,-0.605)	     0.023	       0.023	     0	     0.01
    # j	        (0.06,-0.605)	 0.023	       0.046	     0	     0.01
    shepp_logan = np.zeros(shape)

    # create meshgrid of coords
    yy_base, xx_base = np.mgrid[:shape[0], :shape[1]]

    # center at 0, 0 and normalize
    xx_base = (xx_base - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    yy_base = (yy_base - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)

    # definition of ellipses as np.array:
    el_params = np.array([[0     ,0	      ,0.69	    ,0.92	,0	             ,2     ],
                          [0     ,-0.0184 ,0.6624	,0.874	,0	             ,-0.98 ],
                          [0.22  ,0	      ,0.11	    ,0.31	,np.radians(-18) ,-0.02 ],
                          [-0.22 ,0	      ,0.16	    ,0.41	,np.radians( 18) ,-0.02 ],
                          [0     ,0.35	  ,0.21	    ,0.25	,0	             ,0.01  ],
                          [0     ,0.1     ,0.046	,0.046	,0	             ,0.01  ],
                          [0     ,-0.1    ,0.046	,0.046	,0	             ,0.01  ],
                          [-0.08 ,-0.605  ,0.046	,0.023	,0	             ,0.01  ],
                          [0     ,-0.605  ,0.023	,0.023	,0	             ,0.01  ],
                          [0.06  ,-0.605  ,0.023	,0.046	,0	             ,0.01  ]])

    # create ellipses and sum up
    for i in range(el_params.shape[0]):
        # get params:
        x_pos = el_params[i][0]
        y_pos = el_params[i][1]
        a     = el_params[i][2]
        b     = el_params[i][3]
        phi   = el_params[i][4]
        value = el_params[i][5]

        # move to pos
        xc = (xx_base - x_pos)
        yc = (yy_base - y_pos)
        
        # rotate
        xx = xc * np.cos(phi) + yc * np.sin(phi)
        yy = yc * np.cos(phi) - xc * np.sin(phi)

        # calc squared distance to pos
        ellipse_points = (xx ** 2) / (a ** 2) + (yy ** 2) / (b ** 2)

        # sum up
        shepp_logan = shepp_logan + (ellipse_points <= 1) * value

    return np.flip(shepp_logan, axis=0)


def shepp_logan_enhanced(shape):
    """
        Creates a contrast enhanced Shepp Logan Phantom.

    Args:
        shape: Shape (in [Y, X]) of phantom to create.

    Returns:
        Phantom of shape as np.array

    """
    shepp_logan = np.zeros(shape, dtype=np.float32)

    # create meshgrid of coords
    yy_base, xx_base = np.mgrid[:shape[0], :shape[1]]

    # center at 0, 0 and normalize
    xx_base = (xx_base - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    yy_base = (yy_base - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)

    # definition of ellipses with enhanced contrast values as np.array:
    el_params = np.array([[0     ,0	      ,0.69	    ,0.92	,0	             ,1    ],
                          [0     ,-0.0184 ,0.6624	,0.874	,0	             ,-0.8 ],
                          [0.22  ,0	      ,0.11	    ,0.31	,np.radians(-18) ,-0.2 ],
                          [-0.22 ,0	      ,0.16	    ,0.41	,np.radians( 18) ,-0.2 ],
                          [0     ,0.35	  ,0.21	    ,0.25	,0	             ,0.1  ],
                          [0     ,0.1     ,0.046	,0.046	,0	             ,0.1  ],
                          [0     ,-0.1    ,0.046	,0.046	,0	             ,0.1  ],
                          [-0.08 ,-0.605  ,0.046	,0.023	,0	             ,0.1  ],
                          [0     ,-0.605  ,0.023	,0.023	,0	             ,0.1  ],
                          [0.06  ,-0.605  ,0.023	,0.046	,0	             ,0.1  ]])

    # create ellipses and sum up
    for i in range(el_params.shape[0]):
        # get params:
        x_pos = el_params[i][0]
        y_pos = el_params[i][1]
        a     = el_params[i][2]
        b     = el_params[i][3]
        phi   = el_params[i][4]
        value = el_params[i][5]

        # move to pos
        xc = (xx_base - x_pos)
        yc = (yy_base - y_pos)

        # rotate
        xx = xc * np.cos(phi) + yc * np.sin(phi)
        yy = yc * np.cos(phi) - xc * np.sin(phi)

        # calc squared distance to pos
        ellipse_points = (xx ** 2) / (a ** 2) + (yy ** 2) / (b ** 2)

        # sum up
        shepp_logan = shepp_logan + (ellipse_points <= 1) * value

    return np.flip(shepp_logan, axis=0)


def shepp_logan_3d(shape):
    """
        Creates a Shepp Logan like 3d Phantom. Definition adopted from CONRAD.
    Args:
        shape: Shape (in [Y, X]) of Shepp Logan phantom to create.

    Returns:
        Phantom of shape as np.array
    """
    shepp_logan = np.zeros(shape, dtype=np.float32)

    # create meshgrid of coords
    zz_base, yy_base, xx_base = np.mgrid[:shape[0], :shape[1], :shape[2]]

    # center at 0, 0 and normalize
    xx_base = (xx_base - (shape[2] - 1) / 2) / ((shape[2] - 1) / 2)
    yy_base = (yy_base - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    zz_base = (zz_base - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)

    # definition of ellipsoids as np.array:
    #                       delta_x, delta_y, delta_z,        a,       b,       c,            phi,  theta,  psi,     rho
    el_params = np.array([[       0,       0,       0,     0.69,    0.92,    0.81,              0,      0,    0,     1  ],
                          [       0, -0.0184,       0,   0.6624,   0.874,    0.78,              0,      0,    0,   -0.8 ],
                          [    0.22,       0,       0,     0.11,    0.31,    0.22,   -(np.pi)/10.,      0,    0,   -0.2 ],
                          [   -0.22,       0,       0,     0.16,    0.41,    0.28,    (np.pi)/10.,      0,    0,   -0.2 ],
                          [       0,    0.35,   -0.15,     0.21,    0.25,    0.41,              0,      0,    0,    0.1 ],
                          [       0,     0.1,    0.25,    0.046,   0.046,    0.05,              0,      0,    0,    0.1 ],
                          [       0,    -0.1,    0.25,    0.046,   0.046,    0.05,              0,      0,    0,    0.1 ],
                          [   -0.08,  -0.605,       0,    0.046,   0.023,    0.05,              0,      0,    0,    0.1 ],
                          [       0,  -0.605,       0,    0.023,   0.023,    0.02,              0,      0,    0,    0.1 ],
                          [    0.06,  -0.605,       0,    0.023,   0.046,    0.02,              0,      0,    0,    0.1 ]])

    # create ellipses and sum up
    for i in range(el_params.shape[0]):
        # get params:
        x_pos  = el_params[i][0]
        y_pos  = el_params[i][1]
        z_pos  = el_params[i][2]
        a_axis = el_params[i][3]
        b_axis = el_params[i][4]
        c_axis = el_params[i][5]
        phi    = el_params[i][6]
        value  = el_params[i][9]

        # move to pos
        xc = (xx_base - x_pos)
        yc = (yy_base - y_pos)
        zc = (zz_base - z_pos)

        # Rotation
        c = np.cos(phi)
        s = np.sin(phi)
        Rz_phi   = np.array([[ c, -s,  0 ],
                             [ s,  c,  0 ],
                             [ 0,  0,  1 ]])
        c = np.cos(0)
        s = np.sin(0)
        Ry_theta = np.array([[ c,  0,  s ],
                             [ 0,  1,  0 ],
                             [-s,  0,  c ]])
        c = np.cos(0)
        s = np.sin(0)
        Rz_psi   = np.array([[ c, -s,  0 ],
                             [ s,  c,  0 ],
                             [ 0,  0,  1 ]])

        # R = Rz(phi) * Ry(theta) * Rz(psi)
        R = np.dot(np.dot(Rz_phi, Ry_theta), Rz_psi).T

        xx = xc * R[0, 0] + yc * R[0, 1] + zc * R[0, 2]
        yy = xc * R[1, 0] + yc * R[1, 1] + zc * R[1, 2]
        zz = xc * R[2, 0] + yc * R[2, 1] + zc * R[2, 2]

        # calc squared distance to pos
        ellipse_points = (xx ** 2) / (a_axis ** 2) + (yy ** 2) / (b_axis ** 2) + (zz ** 2) / (c_axis ** 2)

        # sum up
        shepp_logan = shepp_logan + (ellipse_points <= 1) * value

    return shepp_logan

