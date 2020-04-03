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
import datetime

from . import training_parameter as args
from .geometry_parameters import GEOMETRY
from .model import filter_model

class pipeline(object):

    def __init__(self ):
        self.model = filter_model()

        self.results     = dict()
        self.is_init = False

        self.learning_rate = tf.Variable(name='learning_rate', dtype=tf.float32, initial_value=args.LEARNING_RATE, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, epsilon=0.1)

    # @tf.function
    def mse_loss(self, prediction, label):
        mse = tf.keras.losses.mse(prediction, label)
        return tf.reduce_sum(tf.reduce_sum( mse ) )

    # @tf.function
    def ssd_loss(self, prediction, label):
        ssd = (prediction - label)**2
        return tf.reduce_sum( ssd )

    # @tf.function
    def train_step(self, input, label):
        with tf.GradientTape() as tape:
            # tape.watch(self.model.reco)
            predictions = self.model(input)
            loss = self.ssd_loss(predictions, label)
            gradients = tape.gradient(loss , self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.variables))

        self.train_loss(loss)

    # @tf.function
    def validation_step(self, input, label):
        predictions = self.model(input)
        loss = self.ssd_loss(predictions, label)

        self.validation_loss(loss)

    def train(self, inputs_train, labels_train, inputs_validation, labels_validation):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = args.LOG_DIR  + current_time + '/train'
        validation_log_dir = args.LOG_DIR + current_time + '/validation'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)

        train_ds = tf.data.Dataset.from_tensor_slices((inputs_train, labels_train))
        validation_ds = tf.data.Dataset.from_tensor_slices((inputs_validation, labels_validation))

        train_dataset = train_ds.shuffle(len(inputs_train)).batch(args.BATCH_SIZE_TRAIN)
        validation_dataset = validation_ds.batch(args.BATCH_SIZE_VALIDATION)

        # Define our metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)

        loss_epoch_before = 1e16

        self.results.setdefault('initial_filter', self.model.filter_weights.numpy())
        print("Start Training")
        prev_train_loss = 1e15
        prev_val_loss = 1e15
        for epoch in range(1,args.MAX_EPOCHS+1):
            avg_loss = 0

            for (x_train, y_train) in train_dataset:
                self.train_step(x_train, y_train)

            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch)

            for (x_validation, y_validation) in validation_dataset:
                self.validation_step(x_validation, y_validation)

            with validation_summary_writer.as_default():
                tf.summary.scalar('validation_loss', self.validation_loss.result(), step=epoch)

            template = 'Epoch {}, Loss: {}, Validation Loss: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.validation_loss.result()))

            if self.train_loss.result() >= prev_train_loss:
                break
            # if self.validation_loss.result() >= prev_val_loss:
            #     break

            prev_train_loss = self.train_loss.result()
            prev_val_loss = self.validation_loss.result()
            # Reset metrics every epoch
            self.train_loss.reset_states()
            self.validation_loss.reset_states()


            # for step in range(0, len(inputs_train)):
            #     _, loss, reco, current_filter = self.sess.run([self.train_op,self.loss,self.prediction, self.filter_weights])
            #     avg_loss += loss
            #
            # avg_loss = avg_loss / len(inputs_train)
            # if epoch % 20 is 0:
            #     print("epoch: %d | avg epoch loss: %f"%(epoch, avg_loss ))
            #     # Save current model
            #     self.saver.save(self.sess, args.WEIGHTS_DIR)
            # self.validation(epoch)
            #
            # summary = self.sess.run(self.summary, feed_dict={self.avg_loss_placeholder:avg_loss, self.avg_validation_loss_placeholder:self.avg_validation_loss})
            # self.writer.add_summary(summary, epoch )
            #
            # #early stopping if loss is increasing or staying the same after one epoch
            # if avg_loss >= loss_epoch_before:
            #     #Save best model
            #     self.saver.save(self.sess,args.WEIGHTS_DIR)
            #     # break
            #
            # loss_epoch_before = avg_loss
        print("training finshed")
        self.results.setdefault('learned_filter', self.model.filter_weights.numpy())

    def forward(self, input_data, label_data, filter=None):
        a=5
        test_ds = tf.data.Dataset.from_tensor_slices((input_data, label_data)).batch(1)

        if filter is not None:
            self.model.filter_weights.assign(filter)
        #
        # self.sess.run(self.iterator_validation.initializer, feed_dict={self.inputs_validation: input_data, self.labels_validation: label_data})
        # avg_loss = 0
        result = []
        for (x_test, y_test) in test_ds:
            predictions = self.model(x_test)
            loss = self.ssd_loss(predictions,y_test)
            result.append(predictions.numpy())
        #
        # avg_loss/= len(input_data)
        #
        return result
