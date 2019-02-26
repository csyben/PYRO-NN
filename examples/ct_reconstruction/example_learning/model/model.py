import tensorflow as tf
from .geometry_parameters import GEOMETRY
from deep_ct_reconstruction.ct_reconstruction.helpers.filters import filters
from deep_ct_reconstruction.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d

"""
    Here the model in terms of Tensorflow layers is defined.
"""

def forward(images_batch):
    """
    Sets up the network architecture.

    Args:
    images_batch: batch input of training data

    Returns:
    backprojection_layer: The last layer before the loss layer.
    """

    # FFT layer
    fft_layer = tf.cast(tf.spectral.fft(tf.cast(images_batch, dtype=tf.complex64)), tf.complex64)

    # Filtering as multiplication layer
    filter_weights = tf.Variable(tf.convert_to_tensor(filters.ramp(int(GEOMETRY.detector_shape[0])))) # init as ramp filter
    #filter_weights = tf.Variable(tf.convert_to_tensor(np.random.uniform(size=int(GEOMETRY.detector_shape[0])))) # init as random to see something

    filter_layer = tf.multiply(fft_layer, tf.cast(filter_weights, dtype=tf.complex64))

    # IFFT layer
    ifft_layer = tf.cast(tf.spectral.ifft(tf.cast(filter_layer, dtype=tf.complex64)), dtype=tf.float32)

    # Reconstruction Backprojection layer
    backprojection_layer = parallel_backprojection2d(ifft_layer, GEOMETRY)

    return backprojection_layer


def loss(predictions, gt_labels):
    # think about using tf.losses.mean_squared_error(...) here
    l2_loss = tf.reduce_sum(tf.squared_difference(predictions, gt_labels))
    return l2_loss


def training_op(loss, learning_rate):
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