from datetime import datetime
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from pyronn.ct_reconstruction.layers.keras import ParallelBackproj2D, ParallelProj2D
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters import filters
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# Keras layers wrappers for FFT2D, IFFT2D and filter multiplication
class Fft(Layer):
    def __init__(self, **kwargs):
        super(Fft, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Fft, self).build(input_shape)
    def call(self, x):
        spectrum = tf.keras.layers.Lambda(lambda sino: tf.spectral.fft2d(tf.cast(sino, dtype=tf.complex64)))(x)
        real = tf.real(spectrum)
        imag = tf.imag(spectrum)
        return K.permute_dimensions(tf.stack([real, imag]), (1,0,2,3,4))
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape.insert(1, 2)
        return tuple(shape)

class Ifft(Layer):
    def __init__(self, **kwargs):
        super(Ifft, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Ifft, self).build(input_shape)
    def call(self, x):
        x_reordered = K.permute_dimensions(x, (1,0,2,3,4))
        spectrum = tf.complex(x_reordered[0], x_reordered[1])
        return tf.to_float(tf.spectral.ifft2d(spectrum))
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape.pop(1)
        return tuple(shape)

class Filter(Layer):
    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        super(Filter, self).__init__(**kwargs)
    def filter_initializer(self, shape, dtype=None, partition_info=None):
        return filters.ramp(shape) + np.random.rand((shape)) * 0.1
    def build(self, input_shape):
        self.filter_weights = self.add_weight(name='filter', 
                                              shape=self.geometry.detector_shape[0],
                                              initializer=self.filter_initializer, 
                                              trainable=True)
        super(Filter, self).build(input_shape)
    def call(self, x):
        x_ = K.squeeze(x, -1)
        x_ = K.permute_dimensions(x_, (1,0,2,3))
        x_ = tf.complex(x_[0], x_[1])
        fw=filters.ram_lak(800,1)
        filtered = tf.multiply(x_, tf.cast(self.filter_weights, dtype=tf.complex64))
        real = tf.real(filtered)
        imag = tf.imag(filtered)
        return K.expand_dims(K.permute_dimensions(K.stack([real, imag]), (1,0,2,3)))
    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

BATCH_SIZE = 2
NB_ELTS = 2
NB_EPOCHS = 500
detector_shape = 800
detector_spacing = 1
number_of_projections = 360
angular_range = 2*np.pi
source_detector_distance = 1200
source_isocenter_distance = 750
volume_size = 512
volume_shape = [volume_size, volume_size]
volume_spacing = [1,1]
GEOMETRY = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
GEOMETRY.set_ray_vectors(circular_trajectory.circular_trajectory_2d(GEOMETRY))
now = str(datetime.now()).replace(" ","_").split(".")[0].replace(":","")

phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
X = np.repeat((phantom + np.random.rand(volume_size, volume_size) * 0.1)[np.newaxis,:,:, np.newaxis], NB_ELTS, axis=0).astype('float32')
y = np.repeat((phantom)[np.newaxis,:,:, np.newaxis], NB_ELTS, axis=0).astype('float32')

model = tf.keras.Sequential()
model.add(ParallelProj2D(GEOMETRY))
model.add(Fft())
model.add(Filter(GEOMETRY))
model.add(Ifft())
model.add(ParallelBackproj2D(GEOMETRY))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=tf.keras.losses.mse)
model.fit(X, y, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs/' + now)])
model.save_weights('trained_model.h5')

y_ = model.predict(X, batch_size=BATCH_SIZE)
print(f"X.mean: {X.mean()}, y_.mean: {y_.mean()}, y.mean: {y.mean()}")
print(f"X.min: {X.min()}, y_.min: {y_.min()}, y.min: {y.min()}")
print(f"X.max: {X.max()}, y_.max: {y_.max()}, y.max: {y.max()}")
print(f"MSE: {np.mean((y_-y)**2)}")

f,a = plt.subplots(1,3)
a[0].imshow(X[0,:,:,0])
a[1].imshow(y_[0,:,:,0])
a[2].imshow(y[0,:,:,0])
plt.show()
