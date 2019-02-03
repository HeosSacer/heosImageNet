import tensorflow as tf
import numpy as np
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from .models import ModelLoader

cifar10_labels_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                       4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class Validator:
    def __init__(self):

    def load_test_data(self, path=None):
        with open('preprocess_validation.p', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
        return batch

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def main():

        with tf.Session() as sess:
            # First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph('image_classification.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()

            test_features, test_labels = load_test_data()

            x = graph.get_tensor_by_name("input_x:0")
            y = graph.get_tensor_by_name("output_y:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            accuracy = graph.get_tensor_by_name("accuracy:0")

            logits = graph.get_tensor_by_name("logits:0")

            predictions = logits.eval(feed_dict={
                x: test_features,
                keep_prob: 1.
            })

            test_acc = sess.run(accuracy,
                                feed_dict={
                                    x: test_features,
                                    y: test_labels,
                                    keep_prob: 1.
                                })

            print(np.size(predictions, 0))
            print(np.size(test_labels, 0))

            confusion = tf.confusion_matrix(labels=tf.argmax(
                test_labels, 1), predictions=tf.argmax(predictions, 1))

            print(sess.run(confusion))

            print("======Accuracy {} ======".format(test_acc))

            cnf_matrix = confusion_matrix(tf.argmax(test_labels, 1).eval(),
                                          tf.argmax(predictions, 1).eval())

            plot_confusion_matrix(cnf_matrix, classes=cifar10_labels, normalize=True,
                                  title="Normalized confusion matrix. Test Set Accuracy = {}".format(test_acc))

    if __name__ == "__main__":
        main()
