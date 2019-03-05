import pyconrad
import numpy as np


def shepp_logan_3d(shape): # shape is array in Z, Y, X so flip in pyconrad call
    _ = pyconrad.ClassGetter('edu.stanford.rsl.conrad.phantom')
    return _.NumericalSheppLogan3D(*np.flip(shape).tolist()).getNumericalSheppLoganPhantom().as_numpy()


def shepp_logan(shape):
    """
        Creates the Shepp Logan Phantom.

    Args:
        shape: Shape (in [Y, X]) of Shepp Logan phantom to create.

    Returns:
        Shepp Logan of shape as np.array

    """
    # Definition of Shepp Logan Phantom
    # Ellipse	Center	         Major Axis    Minor Axis    Theta   Gray Level
    # a	        (0,0)	         0.69	       0.92	         0	     2
    # b	        (0,−0.0184)	     0.6624	       0.874	     0	     −0.98
    # c	        (0.22,0)	     0.11	       0.31	         −18°	 −0.02
    # d	        (−0.22,0)	     0.16	       0.41	         18°	 −0.02
    # e	        (0,0.35)	     0.21	       0.25	         0	     0.01
    # f	        (0,0.1)	         0.046	       0.046	     0	     0.01
    # g	        (0,−0.1)	     0.046	       0.046	     0	     0.01
    # h	        (−0.08,−0.605)	 0.046	       0.023	     0	     0.01
    # i	        (0,−0.605)	     0.023	       0.023	     0	     0.01
    # j	        (0.06,−0.605)	 0.023	       0.046	     0	     0.01
    shepp_logan = np.zeros(shape)

    # create meshgrid of coords
    yy_base, xx_base = np.mgrid[:shape[0], :shape[1]]

    # center at 0, 0 and normalize
    xx_base = (xx_base - (shape[1]-1)/2) / ((shape[1]-1)/2)
    yy_base = (yy_base - (shape[0]-1)/2) / ((shape[0]-1)/2)

    # definition of ellipses as np.array:
    el_params =    np.array([[0     ,0	     ,0.69	    ,0.92	,0	             ,2     ],
                             [0     ,-0.0184 ,0.6624	,0.874	,0	             ,-0.98 ],
                             [0.22  ,0	     ,0.11	    ,0.31	,np.radians(-18) ,-0.02 ],
                             [-0.22 ,0	     ,0.16	    ,0.41	,np.radians( 18) ,-0.02 ],
                             [0     ,0.35	 ,0.21	    ,0.25	,0	             ,0.01  ],
                             [0     ,0.1     ,0.046	    ,0.046	,0	             ,0.01  ],
                             [0     ,-0.1    ,0.046	    ,0.046	,0	             ,0.01  ],
                             [-0.08 ,-0.605	 ,0.046	    ,0.023	,0	             ,0.01  ],
                             [0     ,-0.605	 ,0.023	    ,0.023	,0	             ,0.01  ],
                             [0.06  ,-0.605	 ,0.023	    ,0.046	,0	             ,0.01  ]])

    for i in range(el_params.shape[0]):
        # get params:
        x_pos = el_params[i][0]
        y_pos = el_params[i][1]
        a     = el_params[i][2]
        b     = el_params[i][3]
        theta = el_params[i][4]
        value = el_params[i][5]

        # move to pos
        xc = (xx_base - x_pos)
        yc = (yy_base - y_pos)

        # rotate
        xx = xc * np.cos(theta) + yc * np.sin(theta)
        yy = yc * np.cos(theta) - xc * np.sin(theta)

        # calc squared distance to pos
        ellipse_points = (xx ** 2) / (a ** 2) + (yy ** 2) / (b ** 2)

        # sum up
        shepp_logan =  shepp_logan + (ellipse_points <= 1) * value

    return np.flip(shepp_logan, axis=0)


def shepp_logan_mod(shape):
    """
        Creates a Modified (better contrast) Shepp Logan Phantom.

    Args:
        shape: Shape (in [Y, X]) of phantom to create.

    Returns:
        Phantom of shape as np.array

    """
    shepp_logan = np.zeros(shape)

    # create meshgrid of coords
    yy_base, xx_base = np.mgrid[:shape[0], :shape[1]]

    # center at 0, 0 and normalize
    xx_base = (xx_base - (shape[1]-1)/2) / ((shape[1]-1)/2)
    yy_base = (yy_base - (shape[0]-1)/2) / ((shape[0]-1)/2)

    # definition of ellipses as np.array:
    el_params =    np.array([[0     ,0	     ,0.69	    ,0.92	,0	             ,1    ],
                             [0     ,-0.0184 ,0.6624	,0.874	,0	             ,-0.8 ],
                             [0.22  ,0	     ,0.11	    ,0.31	,np.radians(-18) ,-0.2 ],
                             [-0.22 ,0	     ,0.16	    ,0.41	,np.radians( 18) ,-0.2 ],
                             [0     ,0.35	 ,0.21	    ,0.25	,0	             ,0.1  ],
                             [0     ,0.1     ,0.046	    ,0.046	,0	             ,0.1  ],
                             [0     ,-0.1    ,0.046	    ,0.046	,0	             ,0.1  ],
                             [-0.08 ,-0.605	 ,0.046	    ,0.023	,0	             ,0.1  ],
                             [0     ,-0.605	 ,0.023	    ,0.023	,0	             ,0.1  ],
                             [0.06  ,-0.605	 ,0.023	    ,0.046	,0	             ,0.1  ]])

    for i in range(el_params.shape[0]):
        # get params:
        x_pos = el_params[i][0]
        y_pos = el_params[i][1]
        a     = el_params[i][2]
        b     = el_params[i][3]
        theta = el_params[i][4]
        value = el_params[i][5]

        # move to pos
        xc = (xx_base - x_pos)
        yc = (yy_base - y_pos)

        # rotate
        xx = xc * np.cos(theta) + yc * np.sin(theta)
        yy = yc * np.cos(theta) - xc * np.sin(theta)

        # calc squared distance to pos
        ellipse_points = (xx ** 2) / (a ** 2) + (yy ** 2) / (b ** 2)

        # sum up
        shepp_logan =  shepp_logan + (ellipse_points <= 1) * value

    return np.flip(shepp_logan, axis=0)