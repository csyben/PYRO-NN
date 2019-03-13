import tensorflow as tf
import os.path
from examples.ct_reconstruction.example_learning.model.geometry_parameters import GEOMETRY
from examples.ct_reconstruction.example_learning.model import model, input_data

# training parameters
LEARNING_RATE          = 0.0001
NUM_TRAINING_SAMPLES   = 100
NUM_VALIDATION_SAMPLES = 50
NUM_TEST_SAMPLES       = 1
MAX_TRAIN_STEPS        = NUM_TRAINING_SAMPLES
MAX_VALID_STEPS        = NUM_VALIDATION_SAMPLES
MAX_TEST_STEPS         = NUM_TEST_SAMPLES
MAX_EPOCHS             = 5

import pyconrad as pyc
import time
pyc.setup_pyconrad()
pyc.start_gui()


class Pipeline:


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
        dataset = tf.data.Dataset.from_tensor_slices((self.data_placeholder, self.labels_placeholder)).batch(1).repeat()

        # create a iterator
        self.iter = dataset.make_initializable_iterator()
        self.images, self.labels = self.iter.get_next()

        # call Model
        self.backprojection_layer = self.model.forward(self.images)
        self.loss                 = self.model.l2_loss(self.backprojection_layer, self.labels)
        self.train_op             = self.model.training_op(self.loss, LEARNING_RATE)

        # summary stuff
        tf.summary.scalar('loss', self.loss)
        self.writer  = tf.summary.FileWriter(self.LOG_DIR)
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    
    def do_model_eval(self, sess, input, labels, batch_size):
        # init with data
        sess.run(self.iter.initializer, feed_dict={self.data_placeholder: input,
                                                       self.labels_placeholder: labels,
                                                       self.batch_size_placeholder: 1})
        for i in range(batch_size):
            reco, loss_value = sess.run ([self.backprojection_layer, self.loss])

        return reco, loss_value


    def run_training(self):
        
        # get data
        train_data_numpy, train_labels_numpy = input_data.generate_training_data(NUM_TRAINING_SAMPLES, 0.05)
        validation_data_numpy, validation_labels_numpy = input_data.generate_validation_data(NUM_VALIDATION_SAMPLES)
        test_data_numpy, test_labels_numpy = input_data.get_test_data(NUM_TEST_SAMPLES)

        # session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            self.build_graph()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            pyc.imshow(np.fft.fftshift(self.model.filter_weights.eval()), 'inital')

            for epoch in range(MAX_EPOCHS):
                print('EPOCH: ', epoch)
                # initialise iterator with train data
                sess.run(self.iter.initializer, feed_dict={self.data_placeholder: train_data_numpy,
                                                           self.labels_placeholder: train_labels_numpy,
                                                           self.batch_size_placeholder: NUM_TRAINING_SAMPLES})
                
                for step in range(MAX_TRAIN_STEPS):
                    # get next batch
                    _, training_loss_value, train_reco = sess.run([self.train_op, self.loss, self.backprojection_layer])

                    #pyc.imshow(train_reco, 'current_train_reco')
                    #time.sleep(1)

                    # print loss
                    if(step % 10 == 0):
                        print('training_loss: ', training_loss_value)
                        sum = sess.run(self.summary)
                        self.writer.add_summary(sum, (epoch * MAX_TRAIN_STEPS) + step)

                    # Save a checkpoint of the model after every epoch.
                    if (step + 1) == MAX_TRAIN_STEPS:
                        print('Saving current model state.')
                        self.saver.save(sess, self.WEIGHTS_DIR, global_step=epoch * MAX_TRAIN_STEPS)

                # Every finished epoch validation
                print('Evaluation on Validation data:')
                sess.run(self.iter.initializer, feed_dict={self.data_placeholder: validation_data_numpy,
                                                           self.labels_placeholder: validation_labels_numpy,
                                                           self.batch_size_placeholder: 1})
                for step in range(MAX_VALID_STEPS):
                    validation_reco, validation_loss_value = sess.run([self.backprojection_layer, self.loss])
                    #pyc.imshow(validation_reco, 'validation_reco')
                    #time.sleep(1)
                    print('validation_loss: ', validation_loss_value)

            # Finished training, eval on test set
            print('-----------------------------------------------')
            print('Run trained model on test data: ')
            sess.run(self.iter.initializer, feed_dict={self.data_placeholder: test_data_numpy,
                                                           self.labels_placeholder: test_labels_numpy,
                                                           self.batch_size_placeholder: 1})
            for step in range(MAX_TEST_STEPS):
                test_reco, test_loss_value = sess.run([self.backprojection_layer, self.loss])
                pyc.imshow(test_reco, 'test_reco')
                #time.sleep(1)
                print('test_loss: ', test_loss_value)
            import numpy as np

            pyc.imshow(np.fft.fftshift(self.model.filter_weights.eval()), 'learned')


            
if __name__ == '__main__':
    p = Pipeline()
    p.run_training()

