import numpy as np
import tensorflow as tf
import lme_custom_ops
import pyconrad as pyc
import matplotlib.pyplot as plt
pyc.setup_pyconrad()


from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters import filters
import pyronn.ct_reconstruction.helpers.misc.generate_sinogram as generate_sinogram


def example_learning_simple():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 200
    volume_shape = [volume_size, volume_size]
    volume_spacing = [0.5, 0.5]

    # Detector Parameters:
    detector_shape = 2*volume_size
    detector_spacing = 0.5

    # Trajectory Parameters:
    number_of_projections = 100
    angular_range = np.pi

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan(volume_shape)
    pyc.imshow(phantom, 'phantom')

    # Training Data
    sinogram = generate_sinogram.generate_sinogram(phantom, parallel_projection2d, geometry)
    pyc.imshow(sinogram, 'sinogram')


    # ------------------ Build Network ------------------

    # Define input sinogram
    input_sinogram_tf = tf.placeholder(tf.float32, shape=geometry.sinogram_shape, name="input_sinogram")

    # FFT layer
    fft_layer = tf.cast(tf.spectral.fft(tf.cast(input_sinogram_tf, dtype=tf.complex64)), tf.complex64)

    # Filtering as multiplication layer
    filter_weights = tf.Variable(tf.convert_to_tensor(filters.ramp(int(geometry.detector_shape[0])))) # init as ramp filter
    #filter_weights = tf.Variable(tf.convert_to_tensor(np.random.uniform(size=int(geometry.detector_shape[0])))) # init as random to see something
    filter_layer = tf.multiply(fft_layer, tf.cast(filter_weights, dtype=tf.complex64))

    # IFFT layer
    ifft_layer = tf.cast(tf.spectral.ifft(tf.cast(filter_layer, dtype=tf.complex64)), dtype=tf.float32)

    # Reconstruction Backprojection layer
    backprojection_layer = parallel_backprojection2d(ifft_layer, geometry)

    # loss function
    ground_truth_tf = tf.placeholder(tf.float32, shape=geometry.volume_shape, name="ground_truth")
    regularizer = tf.nn.l2_loss(filter_weights)
    beta = 0.1
    loss_function = tf.reduce_sum(tf.squared_difference(backprojection_layer, ground_truth_tf))# + tf.cast(beta * regularizer, tf.float32)

    # define tf params
    learning_rate = 1e-6
    iterations = 100
    optimizier = tf.train.AdamOptimizer(learning_rate=learning_rate)

    training_operator = optimizier.minimize(loss_function)

    # showing current filter
    plt.show()

    # ------------------ Training ------------------
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for i in range(iterations):
            # run tf session
            training = sess.run(training_operator, feed_dict={input_sinogram_tf: sinogram, ground_truth_tf: phantom})
            loss_value = sess.run(loss_function,   feed_dict={input_sinogram_tf: sinogram, ground_truth_tf: phantom})

            # show some outputs
            if i%1 is 0:
                # show the current reco
                reco = sess.run(backprojection_layer, feed_dict={input_sinogram_tf: sinogram})
                pyc.imshow(reco, 'reco')

                # show the current filter
                current_filter = filter_weights.eval()
                plt.plot(current_filter)
                plt.ylabel('current_filter')
                plt.pause(0.05)

                # status
                print( "iteration: ", i)
                print( "loss: ", loss_value)

if __name__ == '__main__':
    example_learning_simple()