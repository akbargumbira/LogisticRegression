# coding=utf-8


class LogisticRegression(object):
    """A simple logistic regression implementation with gradient descend.

    This implementation heavily uses the same variables naming as explained in
    Andrew Ng's Machine Learning course
    (https://www.coursera.org/learn/machine-learning/home/welcome).
    """
    def __init__(self, alpha):
        """Constructor of the class.

        :param alpha: The learning rate.
        :type alpha: float
        """
        self.alpha = alpha

