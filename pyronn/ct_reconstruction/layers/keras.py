import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from pyronn.ct_reconstruction.layers.projection_2d import fan_projection2d, parallel_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import fan_backprojection2d, parallel_backprojection2d
from pyronn.ct_reconstruction.layers.projection_3d import cone_projection3d
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d

### Parallel geometry
class ParallelProj2D(Layer):
    """
    Parallel projection wrapper that can be used as a Keras layer.
    Constructor needs the geometry as argument.
    """
    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        super(ParallelProj2D, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ParallelProj2D, self).build(input_shape)
    def call(self, x):
        res_tensor = K.expand_dims(tf.stack(tf.map_fn(lambda im: parallel_projection2d(K.squeeze(im, -1), self.geometry), x)))
        res_tensor.set_shape([None, self.geometry.sinogram_shape[0], self.geometry.sinogram_shape[1], 1])
        return res_tensor
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = self.geometry.sinogram_shape[0]
        shape[2] = self.geometry.sinogram_shape[1]
        return tuple(shape)

class ParallelBackproj2D(Layer):
    """
    Parallel backprojection wrapper that can be used as a Keras layer.
    Constructor needs the geometry as argument.
    """
    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        super(ParallelBackproj2D, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ParallelBackproj2D, self).build(input_shape)
    def call(self, x):
        res_tensor = K.expand_dims(tf.stack(
                        tf.map_fn(lambda sino: parallel_backprojection2d(K.squeeze(sino, -1), self.geometry), x)
                     ))
        res_tensor.set_shape([None, self.geometry.sinogram_shape[0], self.geometry.sinogram_shape[1], 1])
        return res_tensor
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = self.geometry.sinogram_shape[0]
        shape[2] = self.geometry.sinogram_shape[1]
        return tuple(shape)

### Fan geometry
class FanProj2D(Layer):
    """
    2D Fan projection wrapper that can be used as a Keras layer.
    Constructor needs the geometry as argument.
    """
    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        super(FanProj2D, self).__init__(**kwargs)
    def build(self, input_shape):
        super(FanProj2D, self).build(input_shape)
    def call(self, x):
        res_tensor = K.expand_dims(tf.stack(
                        tf.map_fn(lambda im: fan_projection2d(K.squeeze(im, -1), self.geometry), x)
                    ))
        res_tensor.set_shape([None, self.geometry.sinogram_shape[0], self.geometry.sinogram_shape[1], 1])
        return res_tensor
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = self.geometry.sinogram_shape[0]
        shape[2] = self.geometry.sinogram_shape[1]
        return tuple(shape)

class FanBackproj2D(Layer):
    """
    2D Fan backprojection wrapper that can be used as a Keras layer.
    Constructor needs the geometry as argument.
    """
    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        super(FanBackproj2D, self).__init__(**kwargs)
    def build(self, input_shape):
        super(FanBackproj2D, self).build(input_shape)
    def call(self, x):
        res_tensor = K.expand_dims(tf.stack(tf.map_fn(lambda sino: fan_backprojection2d(K.squeeze(sino, -1), self.geometry), x)))
        res_tensor.set_shape([None, self.geometry.sinogram_shape[0], self.geometry.sinogram_shape[1], 1])
        return res_tensor
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = self.geometry.sinogram_shape[0]
        shape[2] = self.geometry.sinogram_shape[1]
        return tuple(shape)

### Cone geometry
class ConeProj3D(Layer):
    """
    3D cone beam projection wrapper that can be used as a Keras layer.
    Constructor needs the geometry as argument.
    """
    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        super(ConeProj3D, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ConeProj3D, self).build(input_shape)
    def call(self, x):
        res_tensor = K.expand_dims(tf.stack(tf.map_fn(lambda im: cone_projection3d(K.squeeze(im, -1), self.geometry), x)))
        res_tensor.set_shape([None, self.geometry.sinogram_shape[0], self.geometry.sinogram_shape[1], 1])
        return res_tensor
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = self.geometry.sinogram_shape[0]
        shape[2] = self.geometry.sinogram_shape[1]
        return tuple(shape)

class ConeBackproj3D(Layer):
    """
    3D cone beam backprojection wrapper that can be used as a Keras layer.
    Constructor needs the geometry as argument.
    """
    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        super(ConeBackproj3D, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ConeBackproj3D, self).build(input_shape)
    def call(self, x):
        res_tensor = K.expand_dims(tf.stack(tf.map_fn(lambda sino: cone_backprojection3d(K.squeeze(sino, -1), self.geometry), x)))
        res_tensor.set_shape([None, self.geometry.sinogram_shape[0], self.geometry.sinogram_shape[1], 1])
        return res_tensor
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = self.geometry.sinogram_shape[0]
        shape[2] = self.geometry.sinogram_shape[1]
        return tuple(shape)
