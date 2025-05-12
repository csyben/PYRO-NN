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


def circle(shape, pos, radius, value=1.0):
    """
        Creates a simple circle primitive.
    Args:
        shape:      Shape (in [Y, X])
        pos:        Center (in [Y, X]) from upper left corner
        radius:     Radius
        value:      Value

    Returns:
        np.array filled with circle
    """
    # create meshgrid of coords
    xx, yy = np.mgrid[:shape[0], :shape[1]].astype(dtype=np.float32)

    # calc squared distance to pos
    circle = (xx - pos[1]) ** 2 + (yy - pos[0]) ** 2

    return (circle <= radius ** 2) *  (np.float32)(value)


def ellipse(shape, pos, half_axes, value=1.0, phi=0.0):
    """
        Creates a simple ellipse primitive.
    Args:
        shape:          Shape (in [Y, X])
        pos:            Center (in [Y, X]) from upper left corner
        half_axes:      Half axes of the ellipse (in [b, a])
        value:          Value
        phi:            Rotation Angle in radians

    Returns:
        np.array filled with ellipse
    """
    # create meshgrid of coords
    xx, yy = np.mgrid[:shape[0], :shape[1]].astype(dtype=np.float32)

    # move to pos
    xc = (xx - pos[1])
    yc = (yy - pos[0])

    # rotate
    xx = xc * np.cos(phi) + yc * np.sin(phi)
    yy = yc * np.cos(phi) - xc * np.sin(phi)

    a = half_axes[1]
    b = half_axes[0]

    # calc squared distance to pos
    ellipse_points = (xx ** 2) / (a ** 2) + (yy ** 2) / (b ** 2)

    return (ellipse_points <= 1) * value


def rect(shape, pos, size, value=1.0):
    """
        Creates a simple rect primitive.
    Args:
        shape:      Shape (in [Y, X])
        pos:        Pos (upper left corner) (in [Y, X]) from upper left corner
        size:       Size  (in [Y, X])
        value:      Value

    Returns:
        np.array filled with rectangle
    """
    # create array and populate it with value
    rectangle = np.zeros(shape, dtype=np.float32)
    rectangle[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1]] = value

    return rectangle

