# coding=utf-8
import numpy as np


class LogisticRegression(object):
    """A simple logistic regression implementation with gradient descend.

    This implementation heavily uses the same variables naming as explained in
    Andrew Ng's Machine Learning course
    (https://www.coursera.org/learn/machine-learning/home/welcome).
    """
    def __init__(self, alpha=0.01):
        """Constructor of the class.

        :param alpha: The learning rate.
        :type alpha: float
        """
        self.alpha = float(alpha)
        self.theta = None

    @staticmethod
    def g(z):
        """The sigmoid function.

        :param z: The input array/number.
        :type z: numpy.ndarray
         """
        return 1/(1 + np.exp(-z))

    def h(self, x):
        """The hypothesis.

        The assumption is that x is already padded with 1 as the prefix.

        :param x: The input data that is already prefixed with 1.
        :type x: numpy.ndarray

        :return: The hypothesis for each data in x.
        :rtype: numpy.ndarray
        """
        net_input = np.dot(x, self.theta)
        return self.g(net_input)

    @staticmethod
    def cost(h, y):
        """The cost for logistic regression.

        :param h: The hypothesis.
        :type h: numpy.ndarray

        :param y: The ground truth label.
        :type y: numpy.ndarray

        :return: The cost for this hypothesis.
        :rtype: numpy.ndarray
        """
        return -y.dot(np.log(h)) - ((1 - y).dot(np.log(1 - h)))

    def fit(self, x, y, epochs, verbose=False):
        """Train the data X with labels y to get the parameter theta.

        :param x: The training data (not prefixed with 1 yet).
        :type x: numpy.ndarray

        :param y: The training label.
        :type y: numpy.ndarray

        :param epochs: The number of epochs for the training.
        :type epochs: int

        :param verbose: A flag to print some stats on every epoch.
        :type verbose: bool
        """
        m = x.shape[0]
        # Add 1 to the x[0]
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        # Random the theta in the range [0, 1]
        self.theta = np.random.rand(x.shape[1])
        for i in range(epochs):
            h = self.h(x)
            self.theta -= (self.alpha/m) * np.dot(x.T, h - y)
            if verbose:
                print 'Cost epoch %s: %s' % (i, self.cost(h, y))

    def predict(self, x):
        """Predict labels for data x.

        :param x: The training data (not prefixed with 1 yet).
        :type x: numpy.ndarray

        :return: The predicted labels of x.
        :rtype: numpy.ndarray
        """
        # Add 1 to the x[0]
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        labels = np.zeros(x.shape[0])
        h = self.h(x)
        labels[np.where(h >= 0.5)] = 1
        return labels

    def evaluate(self, x, y, verbose=True):
        """Returns the accuracy and the confusion matrix of predicting x with the current model.

        :param x: The data.
        :type x: numpy.ndarray

        :param y: The true label.
        :type y: numpy.ndarray

        :param verbose: A flag to print some stats.
        :type verbose: bool

        :return: The accuracy and the confusion matrix.
        :rtype: tuple
        """
        unique_classes = np.unique(y)
        n_class = unique_classes.shape[0]
        predicted_labels = self.predict(x)
        accuracy = float(np.where(predicted_labels == y)[0].shape[0]) / x.shape[0]
        conf = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                conf[i][j] = y[(y == i) & (predicted_labels == j)].shape[0]
        if verbose:
            print 'Accuracy: %s' % accuracy
            print 'Confusion matrix: \n %s' % conf
        return accuracy, conf
