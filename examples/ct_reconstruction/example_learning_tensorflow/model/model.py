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

import tensorflow as tf

from .geometry_parameters import GEOMETRY
from pyronn.ct_reconstruction.helpers.filters.filters import ramp
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from tensorflow.keras import Model

class filter_model(Model):
    """
            Here the model in terms of Tensorflow layers is defined.
        """

    def __init__(self):
        super(filter_model, self).__init__()
        filter = ramp(GEOMETRY.detector_shape[0])
        self.filter_weights = tf.Variable(name='filter_frequency', dtype=tf.float32, initial_value=filter, trainable=True)  # init as ramp filter


    def get_filter(self, sess):
        return sess.run(self.filter_weights)

    def set_filter(self, sess, filter_weights):
        sess.run(self.set_filter_weights, feed_dict={self.filter_weights_placeholder: filter_weights})

    def call(self, x):
        """
                Sets up the network architecture.

                Args:
                images_batch: batch input of training data

                Returns:
                backprojection_layer: The last layer before the loss layer.
                """
        sinogram_frequency = tf.signal.fft(tf.cast(x,dtype=tf.complex64))
        filtered_sinogram_frequency = tf.multiply(sinogram_frequency, tf.cast(self.filter_weights,dtype=tf.complex64))
        filtered_sinogram = tf.math.real(tf.signal.ifft(filtered_sinogram_frequency))
        reco = parallel_backprojection2d(filtered_sinogram, GEOMETRY)
        #tf.nn.relu(reco)
        return reco
