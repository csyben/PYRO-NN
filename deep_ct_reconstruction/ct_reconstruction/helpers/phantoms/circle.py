import numpy as np

import pyconrad as pyc # TODO: get independent of pyconrad
pyc.setup_pyconrad()

def circle(shape, pos, radius, value):

    # create meshgrid of coords
    xx, yy = np.mgrid[:shape[1], :shape[0]]

    # calc squared distance to pos
    circle = (xx - pos[1]) ** 2 + (yy - pos[0]) ** 2

    return (circle <= radius**2) * value

def ellipse(shape, pos, half_axes, value, theta=0):

    # create meshgrid of coords
    xx, yy = np.mgrid[:shape[0], :shape[1]]

    # move to pos
    xx = (xx - pos[1])
    yy = (yy - pos[0])

    # rotate
    xx = np.cos(theta) * xx + np.sin(theta) * yy
    yy = np.sin(theta) * xx - np.cos(theta) * yy

    a = half_axes[1]
    b = half_axes[0]

    # calc squared distance to pos
    ellipse_points = (xx ** 2) / (a**2) + (yy ** 2) / (b**2)

    return (ellipse_points <= 1) * value

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
def shepp_logan_numpy(shape):

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
        xx = (xx_base - x_pos)
        yy = (yy_base - y_pos)

        # rotate
        xx = np.cos(theta) * xx + np.sin(theta) * yy
        yy = np.sin(theta) * xx - np.cos(theta) * yy

        # calc squared distance to pos
        ellipse_points = (xx ** 2) / (a ** 2) + (yy ** 2) / (b ** 2)

        # sum up
        shepp_logan =  shepp_logan + (ellipse_points <= 1) * value

    return np.flip(shepp_logan, axis=0)


if __name__ == '__main__':

    # params
    size_x = 200
    size_y = 300

    # circle
    cricle_phantom = circle([size_y, size_x], [50, 50], 30, 1)
    pyc.imshow(cricle_phantom, 'phantom1')

    # ellipse
    ellipse_phantom = ellipse([200, 200], [50, 50], [11, 31], 1, np.radians(18.0))
    pyc.imshow(ellipse_phantom, 'phantom2')

    # my_shepp_logan
    my_shepp_logan = shepp_logan_numpy([size_y, size_x])
    pyc.imshow(my_shepp_logan, 'my_shepp_logan')

    # pyconrad shepp logan
    _ = pyc.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
    shepp_conrad = _.SheppLogan(size_x, True).as_numpy()
    pyc.imshow(shepp_conrad, 'shepp_conrad')

    # difference
    abs_diff = np.abs(my_shepp_logan - shepp_conrad)
    pyc.imshow(abs_diff, 'diff')
    print('difference: ', np.sum(abs_diff))