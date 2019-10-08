# -*- coding: utf-8 -*-
"""Implementations"""

import numpy as np
import math

def standardize(x):
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x

# taken from labs
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def gradient_descent(loss_function, w, max_iters, gamma, *args, verbose=False):
    # set an optimality stopping criterion
    optimal_g = 1e-4

    # inital evaluation
    f, g = loss_function(w, *args)
    evals = 0

    while True:
        # gradient descent step
        w_new = w - gamma * g

        # compute new loss values
        f_new, g_new = loss_function(w_new, *args)
        evals += 1

        # print progress
        if verbose:
            print("%d - loss: %.3f" % (evals, f_new))

        # update weights / loss / gradient
        w = w_new
        f = f_new
        g = g_new

        # test stopping conditions
        if np.linalg.norm(g, float('inf')) < optimal_g:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optimal_g)
            break

        if evals >= max_iters:
            if verbose:
                print("Reached maximum number of function evaluations %d" % max_iters)
            break

    return w, f


def gradient_descent_linesearch(loss_function, w, max_iters, *args, verbose=False):
    # set an optimality stopping criterion
    linesearch_optTol = 1e-2

    # linesearch param
    linesearch_beta = 1e-4

    # evaluate the initial function value and gradient
    f, g = loss_function(w, *args)
    evals = 0
    gamma = 1.0

    while True:
        # line-search using quadratic interpolation to
        # find an acceptable value of gamma
        gg = g.T.dot(g)

        while True:
            # compute params
            w_new = w - gamma * g
            f_new, g_new = loss_function(w_new, *args)

            evals += 1

            if f_new <= f - linesearch_beta * gamma * gg:
                # we have found a good enough gamma to decrease the loss function
                break

            # update step size
            gamma = (gamma ** 2) * gg/(2. * (f_new - f + gamma * gg))

        # print progress
        if verbose:
            print("%d - loss: %.3f" % (evals, f_new))

        # update step-size for next iteration
        y = g_new - g
        gamma = -gamma * np.dot(y.T,g) / np.dot(y.T,y)

        # safety guards
        if np.isnan(gamma) or gamma < 1e-10 or gamma > 1e10:
            gamma = 1.

        # update weights / loss / gradient
        w = w_new
        f = f_new
        g = g_new

        # test termination conditions
        if np.linalg.norm(g, float('inf')) < linesearch_optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % linesearch_optTol)
            break

        if evals >= max_iters:
            if verbose:
                print("Reached maximum number of function evaluations %d" % max_iters)
            break

    return w, f

def cross_validate(y, tx, train, predict, ratio, n_iter):
    n, d = tx.shape
    n_train = math.floor(ratio * n)
    
    accuracy = 0
    
    for i in range(n_iter):
        shuffle_indices = np.random.permutation(np.arange(n))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
        
        train_y = shuffled_y[:n_train]
        train_tx = shuffled_tx[:n_train,:]
        
        test_y = shuffled_y[n_train:]
        test_tx = shuffled_tx[n_train:,:]
        
        w, loss = train(train_y, train_tx)
        y_pred = predict(test_tx, w)
        
        accuracy += compute_accuracy(y_pred, test_y)
    
    return accuracy / n_iter

def find_max_hyperparam(classifier, lambdas):
    w_best = []
    loss_best = np.inf
    lambda_best = 0
    
    for lambda_ in lambdas:
        w, loss = classifier(lambda_)
        print("Testing hyperparameter value %f - loss: %.3f" % (lambda_, loss))
        if loss < loss_best:
            w_best = w
            loss_best = loss
            lambda_best = lambda_
            
    return lambda_best, w_best, loss_best

def compute_accuracy(ypred, yreal):
    return np.sum(ypred == yreal) / len(yreal)

def least_squares(y, tx):
    n, d = tx.shape
    w = np.zeros(d)

    # solve normal equations
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    # return loss and gradient
    return w, 1/(2*n) * np.sum((y - tx @ w) ** 2)


def ridge_regression(y, tx, lambda_):
    n, d = tx.shape
    w = np.zeros(d)

    # solve normal equations
    w = np.linalg.solve(tx.T @ tx + n*lambda_ * np.eye(d), tx.T @ y)

    # return loss and gradient
    return w, 1/(2*n) * np.sum((y - tx @ w) ** 2) + lambda_/2 * w.T.dot(w)



def least_squares_loss_function(y, tx, w):
    n, d = tx.shape

    # define error vector
    e = y - tx @ w

    # return loss and gradient
    return 1/(2*n) * e.T.dot(e), -1/n * tx.T @ e

def least_squares_GD(y, tx, initial_w, max_iters, gamma=0):
    def loss_function(w):
        # compute the loss and gradient using all examples
        return least_squares_loss_function(y, tx, w)

    if gamma == 0:
        return gradient_descent_linesearch(loss_function, initial_w, max_iters)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma=0):
    def loss_function(w):
        # select a single example to compute the loss and the gradient
        y_sgd, tx_sgd = batch_iter(y, tx, 1)
        return least_squares_loss_function(y_sgd, tx_sgd, w)

    if gamma == 0:
        return gradient_descent_linesearch(loss_function, initial_w, max_iters)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma)



def log_reg_loss_function(y, tx, w):
    n, d = tx.shape
    yXw = y * tx.dot(w)

    # compute the function value
    f = np.sum(np.log(1. + np.exp(-yXw)))

    # compute the gradient value
    res = - y / (1. + np.exp(yXw))
    g = tx.T.dot(res)

    return f, g

def logistic_regression(y, tx, initial_w, max_iters, gamma=0):
    n, d = tx.shape

    def loss_function(w):
        return log_reg_loss_function(y, tx, w)

    if gamma == 0:
        return gradient_descent_linesearch(loss_function, initial_w, max_iters)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma=0):
    n, d = tx.shape

    def loss_function(w):
        f, g = log_reg_loss_function(y, tx, w)

        # add regularization terms
        f += lambda_ / 2. * w.dot(w)
        g += lambda_ * w

        return f, g

    if gamma == 0:
        return gradient_descent_linesearch(loss_function, initial_w, max_iters)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma)
    
    
def kernel_RBF(X1, X2, sigma=1):
    N1, D1 = np.shape(X1)
    N2, D2 = np.shape(X2)
    K = np.zeros((N1, N2))

    for i in range(0, N1):
        for j in range(0, N2):
            K[i, j] = np.sum((X1[i] - X2[j]) ** 2)

    return np.exp(-K / (2 * sigma**2))

def kernel_poly(X1, X2, p=2):
    N1, D1 = np.shape(X1)
    N2, D2 = np.shape(X2)
    return np.power(np.ones((N1, N2)) + X1@X2.T, p)


def kernel_predict(kernel_fun, y, X, Xtest, lambda_=0):
    K = kernel_fun(X, X)
    Ktest = kernel_fun(Xtest, X)
    
    u = np.linalg.solve(K + lambda_ * np.eye(len(y)), y)
    
    return np.sign(Ktest @ u)
