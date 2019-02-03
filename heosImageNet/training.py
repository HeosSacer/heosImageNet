from .image_preprocessing import Image_Preprocessor
from .optimizer import Padam
from .models import *

import pickle
import random
import numpy as np
import tensorflow as tf

OPTIM_PARAMS = {
    'padam': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'p': 0.125,
        'b1': 0.9,
        'b2': 0.999,
        'color': 'darkred',
        'linestyle': '-'
    }
}


class Training:
    def __init__(self, Image_Preprocessor=None, path_to_preprocessed_data=None):
        if path_to_preprocessed_data:
            self.Image_Preprocessor = Image_Preprocessor()
            self.Image_Preprocessor.load_data(path_to_preprocessed_data)
        if Image_Preprocessor:
            self.Image_Preprocessor = Image_Preprocessor
        if not(path_to_preprocessed_data or Image_Preprocessor):
            raise(ValueError)
        self.best_accuracy = 0.00

    def train(self, model=create_fast_net, epochs=150, batch_size=2048, keep_probability=0.7, learning_rate=0.001):
        # load the saved dataset
        valid_features, valid_labels = self.Image_Preprocessor.validation_set

        # Hyper parameters
        global_step = tf.Variable(0, trainable=False)

        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()

        # Inputs
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
        y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Build model conv_net
        heosImageNet = model(x, keep_prob)
        model = tf.identity(heosImageNet, name='heosImageNet')

        # Loss and Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=heosImageNet, labels=y))
        # Init padam Optimizer
        op = OPTIM_PARAMS['padam']
        #optimizer = Padam(learning_rate=learning_rate, p=op['p'], beta1=op['b1'], beta2=op['b2']).minimize(cost)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(heosImageNet, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        # Save Model
        self.saver = tf.train.Saver()

        print('Training...')
        with tf.Session() as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())
            valid_features, valid_labels = self.Image_Preprocessor.validation_set
            # Training cycle
            batch_feature_splits, batch_label_splits = self.Image_Preprocessor.training_set
            batch_feature_splits = self.chunkIt(batch_feature_splits, batch_size)
            batch_label_splits = self.chunkIt(batch_label_splits, batch_size)
            for epoch in range(epochs):
                # Loop over all batches
                for batch_features, batch_labels in zip(batch_feature_splits, batch_label_splits):
                    self.train_neural_network(sess, optimizer, keep_probability,
                                              batch_features, batch_labels, x, y, keep_prob, epoch)

                print('Epoch {:>2}:  '.format(epoch + 1), end='')
                self.print_stats(sess, batch_feature_splits[-1], batch_label_splits[-1],
                                 cost, accuracy, x, y, keep_prob, valid_features, valid_labels)

    def train_neural_network(self, session, optimizer, keep_probability, feature_batch, label_batch, x, y, keep_prob, epoch):
        session.run(optimizer,
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: keep_probability
                    })

    def print_stats(self, session, feature_batch, label_batch, cost, accuracy, x, y, keep_prob, valid_features, valid_labels):
        loss = session.run(cost,
                           feed_dict={
                               x: feature_batch,
                               y: label_batch,
                               keep_prob: 1.
                           })
        valid_acc = session.run(accuracy,
                                feed_dict={
                                    x: valid_features,
                                    y: valid_labels,
                                    keep_prob: 1.
                                })
        print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

        if valid_acc > 0.6 and valid_acc > self.best_accuracy:
            save_model_path = './image_classification'
            save_path = self.saver.save(session, save_model_path)
            self.best_accuracy = valid_acc

    def chunkIt(self, seq, chunk_size):
        avg = len(seq) / float(len(seq)/chunk_size)
        out = []
        last = 0.0
        print("Using {} Batch Size".format(avg))
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out
