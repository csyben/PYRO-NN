import pyronn
import numpy as np
def fibonacci_sphere(n):
    '''
    Calculation of the fibonacci distribution on a unit sphere with n samples.
    :param n: Number of samples on the sphere
    :return: The entered coordinates seperated to x,y,z components
    '''
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = (2 *np.pi * i / goldenRatio)  %(2*np.pi) # to radian  # in range [0°,360°]
    phi = np.arccos(1 - 2*(i+0.5)/n)  #/2  # in range [1°,179°]
    x,z, y = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.vstack([x,y,z]).T

def rotation_matrix_from_points(p1, p2):
    # 计算单位向量
    p1 = np.array(p1) / np.linalg.norm(p1)
    p2 = np.array(p2) / np.linalg.norm(p2)

    # 计算旋转轴（叉积）
    axis = np.cross(p1, p2)
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-5:
        if np.dot(p1, p2) > 0:
            # p1 和 p2 同向，无需旋转
            return np.eye(3)
        else:
            # p1 和 p2 反向，需要旋转180度，找一个垂直于p1的向量作为旋转轴
            axis = np.array([p1[1], -p1[0], 0])
            if np.linalg.norm(axis) == 0:
                axis = np.array([p1[2], 0, -p1[0]])
            axis = axis / np.linalg.norm(axis)
            theta = np.pi
    else:
        axis = axis / axis_length

    # 计算旋转角度（点积）
    cos_theta = np.dot(p1, p2)
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    # 使用罗德里格斯公式（Rodrigues' rotation formula）来计算旋转矩阵
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)

    return R


def fft_and_ifft(sinogram, filter):
    if pyronn.read_backend() == 'torch':
        import torch
        if not isinstance(sinogram, torch.Tensor):
            sinogram = torch.tensor(sinogram).cuda()
        if not isinstance(filter, torch.Tensor):
            filter = torch.tensor(filter).cuda()

        x = torch.fft.fft(sinogram, dim=-1, norm='ortho')
        x = torch.multiply(x, filter)
        x = torch.fft.ifft(x, dim=-1, norm='ortho').real
        return x
    elif pyronn.read_backend() == 'tensorflow':
        import tensorflow as tf
        sino_freq = tf.signal.fft(tf.cast(sinogram, dtype=tf.complex64))
        sino_filtered_freq = tf.multiply(sino_freq, tf.cast(filter, dtype=tf.complex64))
        sinogram_filtered = tf.math.real(tf.signal.ifft(sino_filtered_freq))
        return sinogram_filtered
