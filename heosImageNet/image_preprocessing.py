import pickle
import numpy as np


class Image_Preprocessor:
    def __init__(self):
        self.training_set = []
        self.validation_set = []
        self.test_set = []

    def preprocess(self, data_paths, test_data_path=None, validation_set_size=0.1):
        """
            # Normalizes and One Hot Encodes image data.
            # Saves preprocessed batches as pickle-file.
        """
        if type(data_paths) == type([]):
            n_batches = len(data_paths)
        else:
            n_batches = 1

        valid_features = []
        valid_labels = []
        processed_features = []
        processed_labels = []

        for data_path in data_paths:
            features, labels = self.load_raw_batch(data_path)

            index_of_validation = int(len(features) * validation_set_size)

            p_features, p_labels = self._preprocess(
                features[:-index_of_validation], labels[:-index_of_validation])

            processed_features.extend(p_features)
            processed_labels.extend(p_labels)
            valid_features.extend(features[-index_of_validation:])
            valid_labels.extend(labels[-index_of_validation:])

        # preprocess the all stacked validation dataset
        self.validation_set = self._preprocess(np.array(valid_features), np.array(valid_labels))

        self.training_set = (processed_features, processed_labels)

        if test_data_path:
            self.preprocess_test_data(test_data_path)

    def load_raw_batch(self, path):
        with open(path, mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')

        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']

        return features, labels

    def _preprocess(self, features, labels):
        features = self.normalize(features)
        labels = self.one_hot_encode(labels)
        return (features, labels)

    def normalize(self, x):
        """
            argument
                - x: input image data in numpy array [32, 32, 3]
            return
                - normalized x
        """
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x-min_val) / (max_val-min_val)
        return x

    def one_hot_encode(self, x):
        """
            argument
                - x: a list of labels
            return
                - one hot encoding matrix (number of labels, number of class)
        """
        encoded = np.zeros((len(x), 10))

        for idx, val in enumerate(x):
            encoded[idx][val] = 1

        return encoded

    def preprocess_test_data(self, path):
        # load the test dataset
        with open(path, mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # preprocess the testing data
        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']
        self.test_set = self._preprocess(test_features, test_labels)

    def dump_data(self, path):
        if not(path[-1] == '\\' or path[-1] == '/'):
            path = path + "/"
        pickle.dump(self.training_set, open(path + "training_set.p", 'wb'))
        pickle.dump(self.validation_set, open(path + "validation_set.p", 'wb'))
        pickle.dump(self.test_set, open(path + "training_set.p", 'wb'))

    def load_data(self, path):
        if not(path[-1] == '\\' or path[-1] == '/'):
            path = path + "/"
        self.training_set = pickle.load(path + "training_set.p", encoding='latin1')
        self.validation_set = pickle.load(path + "training_set.p", encoding='latin1')
        self.test_set = pickle.load(path + "training_set.p", encoding='latin1')
