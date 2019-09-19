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
import os

from . import training_parameter as args
from .geometry_parameters import GEOMETRY
from .model import filter_model

class pipeline(object):

    def __init__(self, session ):
        self.sess = session
        self.model = filter_model()

        self.results     = dict()
        self.is_init = False

    def init_placeholder_graph(self):

        self.learning_rate = tf.compat.v1.get_variable(name='learning_rate', dtype=tf.float32, initializer=tf.constant(1e-15), trainable=False)
        self.learning_rate_placeholder = tf.compat.v1.placeholder(tf.float32, name='learning_rate_placeholder')
        self.set_learning_rate = self.learning_rate.assign(self.learning_rate_placeholder)

        self.is_training = tf.compat.v1.get_variable(name="is_training", shape=[], dtype=tf.bool, trainable=False)
        self.set_training = self.is_training.assign(True)
        self.set_validation = self.is_training.assign(False)

        self.avg_loss_placeholder = tf.compat.v1.placeholder(tf.float32, name='avg_loss_placeholder')
        self.avg_validation_loss_placeholder = tf.compat.v1.placeholder(tf.float32, name='avg_validation_loss_placeholder')


    def data_loader(self, inputs, labels):
        # Make pairs of elements. (X, Y) => ((x0, y0), (x1)(y1)),....
        image_set = tf.compat.v1.data.Dataset.from_tensor_slices((inputs, labels))
        # Identity mapping operation is needed to include multi-tthreaded queue buffering.
        image_set = image_set.map(lambda x, y: (x, y), num_parallel_calls=4).prefetch(buffer_size=200)
        # Batch dataset. Also do this if batchsize==1 to add the mandatory first axis for the batch_size
        image_set = image_set.batch(1)
        # Repeat dataset for number of epochs
        image_set = image_set.repeat(args.MAX_EPOCHS+1)
        # Prefetch data to gpu.
        # Select iterator
        iterator = image_set.make_initializable_iterator()
        return iterator


    def build_graph(self):
        #Set Placeholders
        self.init_placeholder_graph()

        #Optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # Tensor placeholders that are initialized later. Input and label shape are assumed to be equal
        self.inputs_train = tf.compat.v1.placeholder(tf.float32, (None, *GEOMETRY.sinogram_shape))
        self.labels_train = tf.compat.v1.placeholder(tf.float32, (None, *GEOMETRY.volume_shape))

        self.inputs_validation = tf.compat.v1.placeholder(tf.float32, (None, *GEOMETRY.sinogram_shape))
        self.labels_validation = tf.compat.v1.placeholder(tf.float32, (None, *GEOMETRY.volume_shape))

        # Get next_element-"operator" and iterator that is initialized later
        self.iterator_train = self.data_loader(self.inputs_train, self.labels_train)
        self.iterator_validation = self.data_loader(self.inputs_validation, self.labels_validation)

        # Get next (batch of) element pair(s)
        self.input_element, self.label_element = tf.cond(self.is_training,
                                                         lambda: self.iterator_train.get_next(),
                                                         lambda: self.iterator_validation.get_next())
        # Generator and loss function

        self.prediction, self.filter_weights = self.model.forward(self.input_element)
        self.loss = self.model.l2_loss(self.prediction, self.label_element)
        self.train_op = optimizer.minimize(self.loss)


        # Summary stuff
        tf.compat.v1.summary.scalar('avg loss', self.avg_loss_placeholder)
        tf.compat.v1.summary.scalar('avg validation loss', self.avg_validation_loss_placeholder)
        self.writer = tf.compat.v1.summary.FileWriter(args.LOG_DIR)
        self.summary = tf.compat.v1.summary.merge_all()
        self.saver = tf.compat.v1.train.Saver()

    def validation(self, epoch):
        #Switch to validation dataset
        self.sess.run(self.set_validation)
        avg_validation_loss = 0
        for step in range(0, args.NUM_VALIDATION_SAMPLES):
            validation_loss, reco, current_filter = self.sess.run([self.loss, self.prediction, self.filter_weights])
            avg_validation_loss=+validation_loss
        self.avg_validation_loss = avg_validation_loss / args.NUM_VALIDATION_SAMPLES
        if epoch % 20 is 0:
            print("epoch: %d | avg validation loss: %f" % (epoch, self.avg_validation_loss))
        # Switch back to training dataset
        self.sess.run(self.set_training)

    def train(self, inputs_train, labels_train, inputs_validation, labels_validation):
        #Setup & initialize graph
        self.build_graph()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())

        #Saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=100)

        #Feed
        self.sess.run(self.iterator_train.initializer, feed_dict={self.inputs_train: inputs_train, self.labels_train: labels_train})
        self.sess.run(self.iterator_validation.initializer, feed_dict={self.inputs_validation: inputs_validation, self.labels_validation: labels_validation})
        loss_epoch_before = 1e16
        _ = self.sess.run([self.set_training, self.set_learning_rate], feed_dict={self.learning_rate_placeholder: args.LEARNING_RATE})
        self.results.setdefault('initial_filter', self.model.get_filter(self.sess))
        print("Start Training")
        for epoch in range(1,args.MAX_EPOCHS+1):
            avg_loss = 0
            for step in range(0, len(inputs_train)):
                _, loss, reco, current_filter = self.sess.run([self.train_op,self.loss,self.prediction, self.filter_weights])
                avg_loss += loss

            avg_loss = avg_loss / len(inputs_train)
            if epoch % 20 is 0:
                print("epoch: %d | avg epoch loss: %f"%(epoch, avg_loss ))
                # Save current model
                self.saver.save(self.sess, args.WEIGHTS_DIR)
            self.validation(epoch)

            summary = self.sess.run(self.summary, feed_dict={self.avg_loss_placeholder:avg_loss, self.avg_validation_loss_placeholder:self.avg_validation_loss})
            self.writer.add_summary(summary, epoch )

            #early stopping if loss is increasing or staying the same after one epoch
            if avg_loss >= loss_epoch_before:
                #Save best model
                self.saver.save(self.sess,args.WEIGHTS_DIR)
                # break

            loss_epoch_before = avg_loss
        print("training finshed")
        self.results.setdefault('learned_filter', self.model.get_filter(self.sess))

    def forward(self, input_data, label_data, filter=None):
        self.sess.run(self.set_validation)

        if filter is not None:
            self.model.set_filter(self.sess,filter)

        self.sess.run(self.iterator_validation.initializer, feed_dict={self.inputs_validation: input_data, self.labels_validation: label_data})
        avg_loss = 0
        result = []
        for step in range(0, len(input_data)):
            loss, reco, current_filter = self.sess.run([self.loss, self.prediction, self.filter_weights])
            avg_loss += loss
            result.append(reco)

        avg_loss/= len(input_data)

        return result, avg_loss
