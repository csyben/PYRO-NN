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
import os.path
from model.geometry_parameters import GEOMETRY
from model import model, input_data, evaluation
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak

# training parameters
LEARNING_RATE          = 1e-6
BATCH_SIZE_TRAIN       = 1
NUM_TRAINING_SAMPLES   = 128
MAX_TRAIN_STEPS        = NUM_TRAINING_SAMPLES//BATCH_SIZE_TRAIN
BATCH_SIZE_VALIDATION  = 1
NUM_VALIDATION_SAMPLES = 10
MAX_VALIDATION_STEPS   = NUM_VALIDATION_SAMPLES//BATCH_SIZE_VALIDATION
NUM_TEST_SAMPLES       = 1
MAX_TEST_STEPS         = NUM_TEST_SAMPLES
MAX_EPOCHS             = 100


class Pipeline:


    def __init__(self):
        self.model = model.Model()
        self.BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
        self.LOG_DIR     = os.path.join(self.BASE_DIR, 'logs/')
        self.WEIGHTS_DIR = os.path.join(self.BASE_DIR, 'trained_models/')
        self.results     = dict()


    def build_graph(self):

        # Placeholders for data, label and batchsize input
        self.batch_size_placeholder = tf.placeholder(tf.int64)
        self.data_placeholder   = tf.placeholder(tf.float32, shape=(None,) + tuple(GEOMETRY.sinogram_shape))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,) + tuple(GEOMETRY.volume_shape))

        # Create tf dataset from placholders
        dataset = tf.data.Dataset.from_tensor_slices((self.data_placeholder, self.labels_placeholder)) \
            .batch(self.batch_size_placeholder).repeat()

        # Create a initializable dataset iterator
        self.iter = dataset.make_initializable_iterator()
        self.sinograms, self.labels = self.iter.get_next()

        # Call model
        self.backprojection_layer = self.model.forward(self.sinograms)
        self.loss                 = self.model.l2_loss(self.backprojection_layer, self.labels)
        self.train_op             = self.model.training_op(self.loss, LEARNING_RATE)

        # Summary stuff
        tf.summary.scalar('loss', self.loss)
        self.writer  = tf.summary.FileWriter(self.LOG_DIR)
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    
    def do_model_eval(self, sess, input, labels, batch_size, steps):

        # Initialize dataset iterator with data
        sess.run(self.iter.initializer, feed_dict={self.data_placeholder: input,
                                                       self.labels_placeholder: labels,
                                                       self.batch_size_placeholder: batch_size})
        # Run model and calculate avg loss for set
        avg_loss = 0
        for i in range(steps):
            reco, loss_value = sess.run ([self.backprojection_layer, self.loss])
            avg_loss +=loss_value

        return reco, loss_value, avg_loss/steps


    def run_training(self):
        
        # Create data
        train_data_numpy, train_labels_numpy           = input_data.generate_training_data  (NUM_TRAINING_SAMPLES, 0)
        validation_data_numpy, validation_labels_numpy = input_data.generate_validation_data(NUM_VALIDATION_SAMPLES)
        test_data_numpy, test_labels_numpy             = input_data.get_test_data           (NUM_TEST_SAMPLES)

        # Session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # Build Graph
            self.build_graph()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Save initial filter for results
            self.results["initial_filter"] = self.model.filter_weights.eval()

            loss_epoch_before = 1e16
            for epoch in range(MAX_EPOCHS):
                print("EPOCH: ", epoch)
                # Initialise dataset iterator with train data
                sess.run(self.iter.initializer, feed_dict={self.data_placeholder: train_data_numpy,
                                                           self.labels_placeholder: train_labels_numpy,
                                                           self.batch_size_placeholder: BATCH_SIZE_TRAIN})
                
                for step in range(MAX_TRAIN_STEPS):
                    # Run next batch
                    _, training_loss_value = sess.run([self.train_op, self.loss])

                    # Print loss all x steps
                    if(step % 50 == 0):
                        print("training_loss: ", training_loss_value/BATCH_SIZE_TRAIN) # training loss of batch
                        summary = sess.run(self.summary)
                        self.writer.add_summary(summary, (epoch * MAX_TRAIN_STEPS) + step)

                    # Save a checkpoint of the model after every epoch.
                    if (step + 1) == MAX_TRAIN_STEPS:
                        print("Saving current model state.")
                        self.saver.save(sess, self.WEIGHTS_DIR, global_step=epoch * MAX_TRAIN_STEPS)

                # early stopping if loss is increasing or staying the same after one epoch
                if training_loss_value >= loss_epoch_before:
                    break
                loss_epoch_before = training_loss_value

                # Every finished epoch do a validation
                print("Evaluation on Validation data:")
                _, _, avg_validation_loss_value = self.do_model_eval(sess, validation_data_numpy, validation_labels_numpy,
                                                                     BATCH_SIZE_VALIDATION, MAX_VALIDATION_STEPS)
                print("avg_validation_loss: ", avg_validation_loss_value)

            # Finished training, eval on test set
            print("-----------------------------------------------")
            print("Run trained model on test data: ")
            self.results["learned_filter_reco_test_data"], self.results["learned_filter_reco_loss_test_data"], _ = \
                self.do_model_eval(sess, test_data_numpy, test_labels_numpy, NUM_TEST_SAMPLES, MAX_TEST_STEPS)
            print("test_loss: ", self.results["learned_filter_reco_loss_test_data"])
            self.results["learned_filter"] = self.model.filter_weights.eval()

            # -----------------------------------------------
            # Generate Cupping results with big circle
            cupping_data_numpy, cupping_labels_numpy = input_data.get_test_cupping_data()

            # Generate learned filter cupping results:
            self.results["learned_filter_reco"], self.results["learned_filter_reco_loss"], _ = \
                self.do_model_eval(sess, cupping_data_numpy, cupping_labels_numpy, 1, 1)

            # Generate Ramp filter cupping results: set the filter weights to ramp and run graph
            self.model.filter_weights.load(self.results["initial_filter"], sess)
            self.results["ramp_reco"], self.results["ramp_reco_loss"], _ = \
                self.do_model_eval(sess, cupping_data_numpy, cupping_labels_numpy, 1, 1)

            # Generate Ram_Lak cupping filter results
            self.model.filter_weights.load(ram_lak(GEOMETRY.detector_shape[-1], GEOMETRY.detector_spacing[-1]), sess)
            self.results["ram_lak_reco"], self.results["ram_lak_reco_loss"], _ = \
                self.do_model_eval(sess, cupping_data_numpy, cupping_labels_numpy, 1, 1)


def plot_results(results):

    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'plots/')
    if not os.path.exists(file_path): os.mkdir(file_path)

    # Result filters:
    ramp_filter    = results["initial_filter"]
    ram_lak_filter = ram_lak(GEOMETRY.detector_shape[-1], GEOMETRY.detector_spacing[-1]) / 4 # divide for correct plot scaling
    learned_filter = results["learned_filter"]

    evaluation.evaluation_filter(ramp_filter, ram_lak_filter, learned_filter, os.path.join(file_path, "filter.png"))

    # Result recos:
    ramp_reco           = results["ramp_reco"]
    ram_lak_reco        = results["ram_lak_reco"] / 4 # divide for correct plot scaling
    learned_filter_reco = results["learned_filter_reco"]

    evaluation.evaluation_three(ramp_reco, ram_lak_reco, learned_filter_reco, GEOMETRY.volume_shape, os.path.join(file_path, "cupping.png"))

            
if __name__ == "__main__":
    the_pipeline = Pipeline()
    the_pipeline.run_training()
    plot_results(the_pipeline.results)
