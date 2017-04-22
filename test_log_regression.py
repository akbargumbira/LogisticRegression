# coding=utf-8
import unittest
import numpy as np
from log_regression import LogisticRegression


class TestLogRegression(unittest.TestCase):
    def setUp(self):
        self.log_regression = LogisticRegression()

    def test_g(self):
        """Test the sigmoid function."""
        input = np.array([-500, 0, 500])
        output = self.log_regression.g(input)
        expected_output = np.array([0, 0.5, 1])
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_h(self):
        """Test the hypothesis function."""
        self.log_regression.theta = np.array([0, 1, -1, 5, -5])
        x = np.array([1, 2, 2, 2, 2])
        output = self.log_regression.h(x)
        self.assertEqual(output, 0.5)

        self.log_regression.theta = np.array([0, 1, -1, 5, -5])
        x = np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 3, 3, 3, 3]])
        output = self.log_regression.h(x)
        np.testing.assert_array_equal(output, np.array([0.5, 0.5, 0.5]))

    def test_fit(self):
        """Test the fit method"""
        # Test with a simple y = x (below the line is labelled 1)
        x_train = np.array(
            [[1, 0.9],
             [2, 2.1],
             [3, 2.9],
             [4, 4.1],
             [5, 4.9],
             [6, 6.1],
             [7, 6.9],
             [8, 8.1],
             [9, 8.9],
             [10, 10.1]
             ])
        y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.log_regression.alpha = 0.1
        self.log_regression.fit(x_train, y_train, 5000)
        # Check the hyphothesis
        x_ = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
        h = self.log_regression.h(x_)
        # The hypothesis in odd index should be >= 0.5
        self.assertTrue(all(item >= 0.5 for item in h[::2]))
        # The hypothesis in even index should be < 0.5
        self.assertTrue(all(item < 0.5 for item in h[1::2]))

    def test_predict(self):
        """Test the prediction"""
        # Test with a simple y = x (below the line is labelled 1)
        x_train = np.array(
            [[1, 0.9],
             [2, 2.1],
             [3, 2.9],
             [4, 4.1],
             [5, 4.9],
             [6, 6.1],
             [7, 6.9],
             [8, 8.1],
             [9, 8.9],
             [10, 10.1]
             ])
        y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        x_test = np.array([[1, 0.9], [2, 2.9], [3, 2.9], [4, 5.9], [5, 4.9]])
        y_test = np.array([1, 0, 1, 0, 1])
        self.log_regression.alpha = 0.1
        self.log_regression.fit(x_train, y_train, 5000)
        prediction = self.log_regression.predict(x_test)
        np.testing.assert_equal(y_test, prediction)
