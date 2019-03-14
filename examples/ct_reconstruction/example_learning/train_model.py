import numpy as np
import tensorflow as tf
import os.path
from examples.ct_reconstruction.example_learning.model.geometry_parameters import GEOMETRY
from examples.ct_reconstruction.example_learning.model import model, input_data

# training parameters
LEARNING_RATE          = 0.0001
BATCH_SIZE_TRAIN       = 10
NUM_TRAINING_SAMPLES   = BATCH_SIZE_TRAIN * 10
MAX_TRAIN_STEPS        = NUM_TRAINING_SAMPLES//BATCH_SIZE_TRAIN
BATCH_SIZE_VALIDATION  = 10
NUM_VALIDATION_SAMPLES = BATCH_SIZE_VALIDATION*10
MAX_VALIDATION_STEPS   = NUM_VALIDATION_SAMPLES//BATCH_SIZE_VALIDATION
NUM_TEST_SAMPLES       = 1
MAX_TEST_STEPS         = NUM_TEST_SAMPLES
MAX_EPOCHS             = 10


class Pipeline:


    # Class as namespace for storing results
    class Results:
        pass


    def __init__(self):
        self.model = model.Model()
        self.BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
        self.LOG_DIR     = os.path.join(self.BASE_DIR, 'logs/')
        self.WEIGHTS_DIR = os.path.join(self.BASE_DIR, 'trained_models/')


    def build_graph(self):
        # placeholders for inputs
        self.batch_size_placeholder = tf.placeholder(tf.int64)
        self.data_placeholder   = tf.placeholder(tf.float32, shape=(None,) + tuple(GEOMETRY.sinogram_shape))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,) + tuple(GEOMETRY.volume_shape))

        # create tf dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.data_placeholder, self.labels_placeholder))\
            .batch(self.batch_size_placeholder).repeat()

        # create a iterator
        self.iter = dataset.make_initializable_iterator()
        self.sinograms, self.labels = self.iter.get_next()

        # call Model
        self.backprojection_layer = self.model.forward(self.sinograms)
        self.loss                 = self.model.l2_loss(self.backprojection_layer, self.labels)
        self.train_op             = self.model.training_op(self.loss, LEARNING_RATE)

        # summary stuff
        tf.summary.scalar('loss', self.loss)
        self.writer  = tf.summary.FileWriter(self.LOG_DIR)
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    
    def do_model_eval(self, sess, input, labels, batch_size, steps):

        # initialize iterator with data
        sess.run(self.iter.initializer, feed_dict={self.data_placeholder: input,
                                                       self.labels_placeholder: labels,
                                                       self.batch_size_placeholder: batch_size})
        # run model and calc avg loss for set
        avg_loss = 0
        for i in range(steps):
            reco, loss_value = sess.run ([self.backprojection_layer, self.loss])
            avg_loss +=loss_value

        return reco, loss_value, avg_loss/steps


    def run_training(self):
        
        # get data
        train_data_numpy, train_labels_numpy           = input_data.generate_training_data  (NUM_TRAINING_SAMPLES, 0.00)
        validation_data_numpy, validation_labels_numpy = input_data.generate_validation_data(NUM_VALIDATION_SAMPLES)
        test_data_numpy, test_labels_numpy             = input_data.get_test_data           (NUM_TEST_SAMPLES)

        # session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # Build Graph
            self.build_graph()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Save initial filter for results
            self.Results.initial_filter = np.fft.fftshift(self.model.filter_weights.eval())

            for epoch in range(MAX_EPOCHS):
                print('EPOCH: ', epoch)
                # initialise iterator with train data
                sess.run(self.iter.initializer, feed_dict={self.data_placeholder: train_data_numpy,
                                                           self.labels_placeholder: train_labels_numpy,
                                                           self.batch_size_placeholder: BATCH_SIZE_TRAIN})
                
                for step in range(MAX_TRAIN_STEPS):
                    # get next batch
                    _, training_loss_value = sess.run([self.train_op, self.loss])

                    # print loss after each batch
                    if(step % BATCH_SIZE_TRAIN == 0):
                        print('training_loss: ', training_loss_value)
                        sum = sess.run(self.summary)
                        self.writer.add_summary(sum, (epoch * MAX_TRAIN_STEPS) + step)

                    # Save a checkpoint of the model after every epoch.
                    if (step + 1) == MAX_TRAIN_STEPS:
                        print('Saving current model state.')
                        self.saver.save(sess, self.WEIGHTS_DIR, global_step=epoch * MAX_TRAIN_STEPS)

                # Every finished epoch validation
                print('Evaluation on Validation data:')
                _, _, avg_validation_loss_value = self.do_model_eval(sess, validation_data_numpy, validation_labels_numpy,
                                                                     BATCH_SIZE_VALIDATION, MAX_VALIDATION_STEPS)
                print('avg_validation_loss: ', avg_validation_loss_value)

            # Finished training, eval on test set
            print('-----------------------------------------------')
            print('Run trained model on test data: ')
            self.Results.test_reco, self.Results.test_loss, _ = self.do_model_eval(sess, test_data_numpy, test_labels_numpy,
                                                                                   NUM_TEST_SAMPLES, MAX_TEST_STEPS)
            print('test_loss: ', self.Results.test_loss)
            self.Results.learned_filter = np.fft.fftshift(self.model.filter_weights.eval())


def plot_results(results):
    pass

            
if __name__ == '__main__':
    p = Pipeline()
    p.run_training()
    plot_results(p.Results)


