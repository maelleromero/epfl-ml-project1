# -*- coding: utf-8 -*-
"""Classifiers"""

import solver
import numpy as np
import math

class least_squares:
    """Class representing the least squares classifier"""

    def __init__(self, verbose=False, max_evaluations=100):
        """
        Constructor

        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """

        self.verbose = verbose
        self.max_evaluations = max_evaluations

    def fit(self, y, X):
        """
        Finds weights to fit the data to the model

        :param y: answers
        :param X: data
        """

        # dimensions
        n, d = X.shape

        # initial weight vector
        self.w = np.zeros(d)

        # find weights
        self.w = np.linalg.solve(X.T @ X, X.T @ y)


    def function_object(self, w, y, X):
        """
        Function Object.

        :param y: answers
        :param X: data
        :param w: weights
        :return: loss, gradient
        """

        # dimensions
        n, d = X.shape

        # compute error
        e = y - X @ w

        # compute loss
        f = 1/(2 * n) * np.sum(e ** 2)

        # compute gradient
        g = - 1 / n * X.T @ e

        return f, g

    def predict(self, X):
        """
        Predict

        :param X: data
        :return: answer prediction
        """

        return np.sign(X @ self.w)

class logistic_regression:
    """Logistic Regression"""

    def __init__(self, verbose=False, max_evaluations=100):
        """
        Constructor

        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """

        self.verbose = verbose
        self.max_evaluations = max_evaluations

    def fit(self, y, X):
        """
        Finds weights to fit the data to the model

        :param y: answers
        :param X: data
        """

        # dimensions
        n, d = X.shape

        # initial weight vector
        self.w = np.zeros(d)

        # fit weights
        self.w, f = solver.gradient_descent(self.function_object, self.w, self.max_evaluations,
                                 y, X, verbose=self.verbose)

    def sigmoid(t):
        """
        Sigmoid

        :param t: parameter
        :return: apply sigmoid function on t
        """
        return 1.0 / (1 + np.exp(- t))

    def function_object(self, w, y, X):
        """
        Function Object.

        :param y: answers
        :param X: data
        :param w: weights
        :return: loss, gradient
        """

        pred = self.sigmoid(X.dot(w))
        f = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        g = X.T.dot(pred - y)

        return np.squeeze(- f), g

class logistic_regression_L2(logistic_regression):
    """L2 Regularized Logistic Regression"""

    def __init__(self, lammy=1.0, verbose=False, max_evaluations=100):
        """
        Constructor

        :param lammy: lambda of L2 regularization
        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """
        super(logistic_regression_L2, self).__init__(verbose=verbose,
                                                     max_evaluations=max_evaluations)
        self.lammy = lammy

    def funObj(self, w, y, X):
        """
        Function Object

        :param w: weight
        :param y: answers
        :param X: data
        :return: loss, gradient
        """
        # Obtain normal loss and gradient using the superclass
        f, g = super(logistic_regression_L2, self).funObj(w, y, X)

        # Add L2 regularization
        f += self.lammy / 2. * w.dot(w)
        g += self.lammy * w

        return f, g