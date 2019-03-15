import tensorflow as tf
from .geometry_parameters import GEOMETRY
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d


class Model:
    """
        Here the model in terms of Tensorflow layers is defined.
    """

    def __init__(self):
        pass


    def forward(self, sinogram_batch):
        """
        Sets up the network architecture.

        Args:
        images_batch: batch input of training data

        Returns:
        backprojection_layer: The last layer before the loss layer.
        """

        # FFT layer
        fft_layer = tf.spectral.fft(tf.cast(sinogram_batch, dtype=tf.complex64))

        # Filtering as multiplication layer
        self.filter_weights = tf.Variable(initial_value=filters.ramp(GEOMETRY.detector_shape[0]), expected_shape=GEOMETRY.detector_shape[0]) # init as ramp filter
        #filter_weights = tf.Variable(initial_value=tf.random.uniform(GEOMETRY.detector_shape[0]), expected_shape=GEOMETRY.detector_shape[0]) # init as random

        filter_layer = tf.multiply(fft_layer, tf.cast(self.filter_weights, dtype=tf.complex64))

        # IFFT layer
        ifft_layer = tf.real(tf.spectral.ifft(filter_layer))

        # Reconstruction Backprojection layer
        self.backprojection_layer = tf.nn.relu(parallel_backprojection2d(ifft_layer, GEOMETRY))

        return self.backprojection_layer


    def l2_loss(self, predictions, gt_labels):
        self.loss = tf.reduce_sum(tf.squared_difference(predictions, gt_labels))
        return self.loss


    def training_op(self, loss, learning_rate):
        """
            Sets up the training Ops.

            Args:
            loss: Loss tensor, from loss().
            learning_rate: The learning rate to use for gradient descent.

            Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op
