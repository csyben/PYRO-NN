import os.path
import tensorflow as tf

from examples.ct_reconstruction.example_learning.model import input_data
from examples.ct_reconstruction.example_learning.model import model
from examples.ct_reconstruction.example_learning.model.geometry_parameters import GEOMETRY

# training parameters
LEARNING_RATE       = 0.0001
BATCH_SIZE          = 50
MAX_TRAIN_STEPS     = 800
MAX_VALID_STEPS     = 200
MAX_TEST_STEPS      = 200
MAX_EPOCHS          = 15

def run_training():
    
    BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
    LOG_DIR     = os.path.join(BASE_DIR, 'logs/')
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'trained_models/')

    # get training data
    train_data_np, labels_np = input_data.generate_training_data(BATCH_SIZE)

    with tf.Graph().as_default():

        # placeholders for inputs
        data_ph   = tf.placeholder(tf.float32, shape=train_data_np.shape)
        labels_ph = tf.placeholder(tf.float32, shape=labels_np.shape)

        # create tf dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((data_ph, labels_ph)).batch(BATCH_SIZE).repeat()

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        images, labels = iter.get_next()

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_dataset)

        # call Model
        output   = model.forward(images)
        loss     = model.loss(output, labels)
        train_op = model.training_op(loss, LEARNING_RATE)

        # summary stuff
        tf.summary.scalar('loss', loss)
        writer  = tf.summary.FileWriter(LOG_DIR)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver()

        # session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # initialise iterator with train data
        sess.run(train_init_op, feed_dict={data_ph: train_data_np, labels_ph: labels_np})

        for epoch in range(MAX_EPOCHS):

            print('EPOCH : ', epoch)
            
            for step in range(MAX_TRAIN_STEPS):

                # get next batch
                _, loss_value = sess.run([train_op, loss])

                # print loss
                if(step % 100 == 0):
                    print('loss: ', loss_value)
                    sum = sess.run(summary)
                    writer.add_summary(sum, (epoch * MAX_TRAIN_STEPS) + step)

                # Save a checkpoint of the model after every epoch.
                if (step + 1) == MAX_TRAIN_STEPS:
                    print('Saving current model state')
                    saver.save(sess, WEIGHTS_DIR, global_step=epoch * MAX_TRAIN_STEPS)

            # Every finished epoch validation
            #print('Evaluation on Validation data:')
            # TODO: every finsihed epoch validation and image of current filter + reco

        # TODO: testing at end of training
        # Finished training, eval on test set
        #print('-----------------------------------------------')
        #print('Run trained model on test data: ')

            
if __name__ == '__main__':
    run_training()
