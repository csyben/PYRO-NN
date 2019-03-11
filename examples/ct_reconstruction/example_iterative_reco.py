import numpy as np
import tensorflow as tf
import argparse

from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers import projection_2d
#TODO: Remove PyConrad
import pyconrad as pyc
pyc.setup_pyconrad()

def iterative_reconstruction():
    # ------------------ Declare Parameters ------------------

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-2, help='initial learning rate for adam')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=1000000, help='# of epoch')
    args = parser.parse_args()

    # Volume Parameters:
    volume_size = 512
    volume_shape = [volume_size, volume_size]
    volume_spacing = [0.5, 0.5]

    # Detector Parameters:
    detector_shape = 625
    detector_spacing = 0.5

    # Trajectory Parameters:
    number_of_projections = 30
    angular_range = np.radians(200)  # 200 * np.pi / 180

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geometry))

    phantom = shepp_logan.shepp_logan_mod(volume_shape)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config=config) as sess:
        acquired_sinogram = generate_sinogram(phantom,projection_2d.parallel_projection2d,geometry)

        acquired_sinogram = acquired_sinogram + np.random.normal(
            loc=np.mean(np.abs(acquired_sinogram)), scale=np.std(acquired_sinogram), size=acquired_sinogram.shape) * 0.02

        zero_vector = np.zeros(np.shape(phantom), dtype=np.float32)

        iter_pipeline = pipeline(sess, args, geometry)
        iter_pipeline.train(zero_vector,np.asarray(acquired_sinogram))
        pyc.imshow(iter_pipeline.result, 'result reco')

########

class pipeline(object):

    def __init__(self, session, args, geometry):
        self.args = args
        self.sess = session
        self.geometry = geometry
        self.model = iterative_reco_model(geometry, np.zeros(geometry.volume_shape, dtype=np.float32))
        self.regularizer_weight = 0.5

    def init_placeholder_graph(self):
        self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, initializer=tf.constant(0.0001), trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate_placeholder')
        self.set_learning_rate = self.learning_rate.assign(self.learning_rate_placeholder)


    def build_graph(self, input_type, input_shape, label_shape):

        self.init_placeholder_graph()
        g_opt = tf.train.AdamOptimizer(self.learning_rate)

        # Tensor placeholders that are initialized later. Input and label shape are assumed to be equal
        self.input_placeholder = tf.placeholder(input_type, (None, input_shape[0], input_shape[1]))
        self.label_placeholder = tf.placeholder(input_type, (None, label_shape[0], label_shape[1]))

        # Make pairs of elements. (X, Y) => ((x0, y0), (x1)(y1)),....
        image_set = tf.data.Dataset.from_tensor_slices((self.input_placeholder, self.label_placeholder))
        # Identity mapping operation is needed to include multi-tthreaded queue buffering.
        image_set = image_set.map(lambda x, y: (x, y), num_parallel_calls=4).prefetch(buffer_size=200)
        # Batch dataset. Also do this if batchsize==1 to add the mandatory first axis for the batch_size
        image_set = image_set.batch(1)
        # Repeat dataset for number of epochs
        image_set = image_set.repeat(self.args.num_epochs + 1)
        # Select iterator
        self.iterator = image_set.make_initializable_iterator()

        self.input_element, self.label_element  = self.iterator.get_next()

        self.current_sino, self.current_reco = self.model.model(self.input_element)

        tv_loss_x = tf.image.total_variation(tf.transpose(self.current_reco))
        tv_loss_y = tf.image.total_variation(self.current_reco)

        self.loss = tf.reduce_sum(tf.squared_difference(self.label_element, self.current_sino)) + self.regularizer_weight*(tv_loss_x+tv_loss_y)
        self.train_op = g_opt.minimize(self.loss)


    def train(self, zero_vector, acquired_sinogram):
        self.build_graph(zero_vector.dtype, zero_vector.shape, acquired_sinogram.shape)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        learning_rate = self.args.learning_rate

        zero_vector_train = np.expand_dims(zero_vector, axis=0)
        acquired_sinogram_train = np.expand_dims(acquired_sinogram, axis=0)

        # initialise iterator with train data
        #self.sess.run(self.train_init_op, feed_dict={self.input_placeholder: zero_vector_train, self.label_placeholder: acquired_sinogram_train})
        self.sess.run(self.iterator.initializer, feed_dict={self.input_placeholder: zero_vector_train, self.label_placeholder: acquired_sinogram_train})

        min_loss = 10000000000000000
        for epoch in range(1, self.args.num_epochs + 1):

            _ = self.sess.run([self.set_learning_rate], feed_dict={self.learning_rate_placeholder: learning_rate})

            _, loss, current_sino, current_reco, label = self.sess.run([self.train_op, self.loss, self.current_sino, self.current_reco, self.label_element])

            if loss > min_loss * 1.005:
                break
            if epoch % 50 is 0:
                print('Epoch: %d' % epoch)
                print('Loss %f' % loss)
                pyc.imshow(current_reco, 'Phantom: Current Reco')
            if min_loss > loss:
                min_loss = loss
                self.result = current_reco

class iterative_reco_model:

    def __init__(self, geometry, reco_initialization):
        self.geometry = geometry
        self.reco = tf.get_variable(name='reco', dtype=tf.float32,
                                    initializer=tf.expand_dims(reco_initialization, axis=0),
                                    trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    def model(self, input_volume):
        self.updated_reco = tf.add(input_volume, self.reco)
        self.current_sino = projection_2d.parallel_projection2d(self.updated_reco, self.geometry)
        return self.current_sino, self.reco





def medphys_plot():
    import matplotlib.pyplot as plt
    import pyconrad as pyc
    pyc.setup_pyconrad()
    path = '/home/syben/Documents/medphys/'
    _ = pyc.ClassGetter()
    phantom = pyc.PyGrid.from_tiff(path+'phantom_2d.tif')
    iter = pyc.PyGrid.from_tiff(path + 'ops_iter_reco.tif')
    parallel = pyc.PyGrid.from_tiff(path + 'ops_parallel_reco.tif')

    slice = 128
    row = 256

    phantom_central = np.array(phantom[0, :, :])
    custom_ops_central = np.array(iter[0, :, :])
    conrad_central = np.array(parallel[0, :, :])

    gt = phantom_central[row,:]
    p1 = custom_ops_central[row,:]
    p2 = conrad_central[row,:]

    fig = plt.figure(figsize=(40, 10))
    number_of_collumns = 8
    number_of_rows = 3
    img_one = plt.subplot2grid((number_of_rows, number_of_collumns), (0, 0), colspan=2,rowspan=3)
    img_two = plt.subplot2grid((number_of_rows, number_of_collumns), (0, 2), colspan=2, rowspan=3)

    plot_one = plt.subplot2grid((number_of_rows, number_of_collumns), (0, 4), colspan=4, rowspan=3)  # , sharex=img_one)


    img_width = np.shape(phantom_central)[0]
    img_one.plot((0, img_width), (row, row), '--', linewidth=5)
    img_two.plot((0, img_width), (row, row), '--', linewidth=5)

    img_one.imshow(custom_ops_central, cmap=plt.get_cmap('gist_gray'), vmin=0, vmax=1)
    img_one.axis('off')
    img_one.set_title('Iterative NN Reconstruction', fontsize=40, y=1.05)

    img_two.imshow(conrad_central, cmap=plt.get_cmap('gist_gray'), vmin=0, vmax=1)
    img_two.axis('off')
    img_two.set_title('FBP NN Reconstruction', fontsize=40, y=1.05)

    plot_one.plot(np.arange(len(gt)), gt, color='#1f77b4')
    plot_one.plot(np.arange(len(p1)), p1, color='lightgreen')
    plot_one.plot(np.arange(len(p2)), p2, linestyle=':', color='red')

    min_plt = 0
    max_plt = 1.1

    plot_one.set_ylim(plt.ylim((min_plt, max_plt)))
    plot_one.legend(['Shepp-Logan Phantom', 'NN Reconstruction', 'FBP NN Reconstruction'], loc='upper center', prop={'size': 40})
    #plt.savefig('evaluation/eval_img/' + experiment_img_name + '.png', dpi=150, transparent=False, bbox_inches='tight')
    plt.gcf().tight_layout(h_pad=0, w_pad=0)

    plt.savefig(path + 'iter_eval.png', dpi=150, transparent=False, bbox_inches='tight')

    fig.show()

    a = 5


if __name__ == '__main__':
    #medphys_plot()
    iterative_reconstruction()