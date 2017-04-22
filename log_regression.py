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
        :type z: list, int
         """
        return 1/(1 + np.exp(-z))

    def h(self, x):
        """The hypothesis.

        The assumption is that x is already padded with 1 as the prefix.
        """
        net_input = np.dot(x, self.theta)
        return self.g(net_input)

    @staticmethod
    def cost(h, y):
        """The cost for logistic regression."""
        return -y.dot(np.log(h)) - ((1 - y).dot(np.log(1 - h)))

    def fit(self, x, y, epochs, verbose=False):
        """Train the data X with labels y."""
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
        """Predict labels for data x."""
        # Add 1 to the x[0]
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        labels = np.zeros(x.shape[0])
        h = self.h(x)
        labels[np.where(h >= 0.5)] = 1
        return labels

