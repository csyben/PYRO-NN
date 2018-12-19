import pyconrad
import numpy as np


def shepp_logan(xy):
    _ = pyconrad.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
    return _.SheppLogan(xy, False).as_numpy()

def shepp_logan_3d(shape): # shape is array in Z, Y, X so flip in pyconrad call
    _ = pyconrad.ClassGetter('edu.stanford.rsl.conrad.phantom')
    return _.NumericalSheppLogan3D(*np.flip(shape).tolist()).getNumericalSheppLoganPhantom().as_numpy()

