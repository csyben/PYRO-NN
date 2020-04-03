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
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras import Model

from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers import projection_2d


def iterative_reconstruction():
    # ------------------ Declare Parameters ------------------

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-2, help='initial learning rate for adam')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=100000, help='# of epoch')
    args = parser.parse_args()

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size-1, volume_size]#, volume_size+1]
    volume_spacing = [1,1]#, 1]

    # Detector Parameters:
    detector_shape = [375]#, 375]
    detector_spacing = [1]#,1]

    # Trajectory Parameters:
    number_of_projections = 30
    angular_range = np.radians(200)  # 200 * np.pi / 180

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)#, 1200.0, 750.0 )
    matrices = circular_trajectory.circular_trajectory_2d(geometry)

    geometry.set_trajectory(matrices)
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape).astype(dtype=np.float32)
    # Add required batch dimension
    phantom = np.expand_dims(phantom,axis=0)

    # ------------------ Call Layers ------------------

    acquired_sinogram = generate_sinogram(phantom,projection_2d.parallel_projection2d,geometry)

    # acquired_sinogram = acquired_sinogram + np.random.normal(
    #     loc=np.mean(np.abs(acquired_sinogram)), scale=np.std(acquired_sinogram), size=acquired_sinogram.shape) * 0.02

    zero_vector = np.zeros(np.shape(phantom), dtype=np.float32)

    iter_pipeline = pipeline(args, geometry)
    iter_pipeline.train(zero_vector,np.asarray(acquired_sinogram))

    plt.figure()
    plt.imshow(np.squeeze(iter_pipeline.result), cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig('iter_tv_reco.png', dpi=150, transparent=False, bbox_inches='tight')


########

class pipeline(object):

    def __init__(self, args, geometry):
        self.args = args
        self.geometry = geometry
        self.model = iterative_reco_model(geometry, np.zeros(geometry.volume_shape, dtype=np.float32))
        self.regularizer_weight = 0.0001
        self.learning_rate = tf.Variable(name='learning_rate', dtype=tf.float32, initial_value= self.args.learning_rate, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    # @tf.function
    def loss(self, prediction, label, regularizer = False):
        mse = tf.keras.losses.mse(prediction, label)
        tv_loss = 0
        if regularizer:
            tv_loss = tf.image.total_variation(prediction)
        return tf.reduce_sum(tf.reduce_sum( mse ) + self.regularizer_weight * tv_loss)
    # @tf.function
    def train_step(self, input, label):
        with tf.GradientTape() as tape:
            # tape.watch(self.model.reco)
            predictions, current_reco = self.model(input)
            self.loss_v = self.loss(predictions, label, False)
            gradients = tape.gradient(self.loss_v , self.model.trainable_variables)
        # print('loss: ',self.loss_v.numpy())
        self.optimizer.apply_gradients(zip(gradients, self.model.variables))


    def train(self, zero_vector, acquired_sinogram):
        import pyconrad as pyc
        pyc.setup_pyconrad()
        self.data_iterator = tf.data.Dataset.from_tensor_slices((zero_vector, acquired_sinogram)).batch(1)

        last_loss = 100000000
        for epoch in range(self.args.num_epochs):
            for images, labels in self.data_iterator:
                self.train_step(images, labels)
            if epoch % 25 is 0:
                pyc.imshow(self.model.reco.numpy(), 'reco')
            if epoch % 100 is 0:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch, self.loss_v.numpy()))
            if self.loss_v.numpy() > last_loss*1.03:
                print('break at epoch', epoch)
                break
            last_loss = self.loss_v.numpy()

        print('training finished')
        self.result = self.model.reco.numpy()



class iterative_reco_model(Model):

    def __init__(self, geometry, reco_initialization):
        super(iterative_reco_model, self).__init__()
        self.geometry = geometry
        self.reco = tf.Variable(name='reco', dtype=tf.float32,
                                    initial_value=tf.expand_dims(reco_initialization, axis=0),
                                    trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    def call(self, x):
        self.updated_reco = tf.add(x, self.reco)
        self.current_sino = projection_2d.parallel_projection2d(self.updated_reco, self.geometry)
        return self.current_sino, self.reco


if __name__ == '__main__':
    iterative_reconstruction()