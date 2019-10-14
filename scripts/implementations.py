# -*- coding: utf-8 -*-
"""Implementations"""

import numpy as np
import math

def standardize(x):
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x

# removes features from the data from string input
# also removes wanted features from features list
# data is tX
# features is list of features from load_csv
# feats is array of strings of features that we want to remove
def remove_features(data, features, feats, verbose=False):

    idx_to_remove = np.ones(len(feats))
    removed = []

    for i, feat in enumerate(feats):
        idx_to_remove[i] = np.where(features == feat)[0]
        if not (idx_to_remove[i].shape == 0):
            removed.append(feat)

    idx_to_remove = idx_to_remove.astype(np.int)

    if verbose:
        print("Features removed:", *removed, sep='\n')

    return np.delete(data, idx_to_remove, 1), np.delete(features, idx_to_remove)

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
    gamma = 1.

    while True:
        # line-search using quadratic interpolation to
        # find an acceptable value of gamma
        gg = g.T.dot(g)
        w_evals = 0

        while True:
            # compute params
            w_new = w - gamma * g
            f_new, g_new = loss_function(w_new, *args)

            w_evals += 1
            evals += 1

            if f_new <= f - linesearch_beta * gamma * gg:
                # we have found a good enough gamma to decrease the loss function
                break
                
            if verbose:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # update step size
            if np.isinf(f_new):
                gamma = 1. / (10**w_evals)
            else:
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

def gradient_descent_sparse(loss_function, w, lambda_, max_iters, *args, verbose=False):
    """
    Uses the L1 proximal gradient descent to optimize the objective function

    The line search algorithm divides the step size by 2 until
    it find the step size that results in a decrease of the L1 regularized
    objective function
    """
    # parameters of the optimization
    linesearch_optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = loss_function(w,*args)
    evals = 1

    alpha = 1.
    proxL1 = lambda w, alpha: np.sign(w) * np.maximum(abs(w) - lambda_*alpha,0)
    L1Term = lambda w: lambda_ * np.sum(np.abs(w))

    while True:
        gtd = None
        # start line search to determine alpha
        while True:
            w_new = w - alpha * g
            w_new = proxL1(w_new, alpha)

            if gtd is None:
                gtd = g.T.dot(w_new - w)

            f_new, g_new = loss_function(w_new, *args)
            evals += 1

            if f_new + L1Term(w_new) <= f + L1Term(w) + gamma*alpha*gtd:
                # Wolfe condition satisfied, end the line search
                break

            if verbose > 1:
                print("Backtracking... f_new: %.3f, f: %.3f" % (f_new, f))

            # update alpha
            alpha /= 2.

        # print progress
        if verbose > 0:
            print("%d - alpha: %.3f - loss: %.3f" % (evals, alpha, f_new))

        # update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)

        # safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        # update weights / loss / gradient
        w = w_new
        f = f_new
        g = g_new

        # test termination conditions
        opt_cond = np.linalg.norm(w - proxL1(w - g, 1.0), float('inf'))

        if opt_cond < linesearch_optTol:
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

    # return weights and loss
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
    
def least_squares_sparse(y, tx, lambda_, initial_w, max_iters):
    def loss_function(w):
        # compute the loss and gradient using all examples
        return least_squares_loss_function(y, tx, w)

    return gradient_descent_sparse(loss_function, initial_w, lambda_, max_iters)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma=0):
    def loss_function(w):
        # select a single example to compute the loss and the gradient
        ySGD = []
        xSGD = []
        for y_sgd, tx_sgd in batch_iter(y, tx, 1):
            ySGD = y_sgd
            xSGD = tx_sgd
            break
        return least_squares_loss_function(ySGD, xSGD, w)

    if gamma == 0:
        return gradient_descent_linesearch(loss_function, initial_w, max_iters)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma)



def log_reg_loss_function(y, tx, w):
    n, d = tx.shape
    yXw = y * (tx @ w)

    # compute the function value
    f = np.sum(np.log(1. + np.exp(-yXw)))

    # compute the gradient value
    g = tx.T @ (- y / (1. + np.exp(yXw)))

    return f, g

def logistic_regression(y, tx, initial_w, max_iters, gamma=0, verbose=False):
    def loss_function(w):
        return log_reg_loss_function(y, tx, w)

    if gamma == 0:
        return gradient_descent_linesearch(loss_function, initial_w, max_iters, verbose=verbose)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma, verbose=verbose)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma=0, verbose=False):
    def loss_function(w):
        f, g = log_reg_loss_function(y, tx, w)

        # add regularization terms
        f += lambda_ / 2. * w.dot(w)
        g += lambda_ * w

        return f, g

    if gamma == 0:
        return gradient_descent_linesearch(loss_function, initial_w, max_iters, verbose=verbose)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma, verbose=verbose)
    
def logistic_regression_sparse(y, tx, lambda_, initial_w, max_iters, verbose=False):
    def loss_function(w):
        return log_reg_loss_function(y, tx, w)

    return gradient_descent_sparse(loss_function, initial_w, lambda_, max_iters, verbose=verbose)
    
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


def kernel_predict(kernel_fun, y, X, Xtest, *args, lambda_=0):
    K = kernel_fun(X, X, *args)
    Ktest = kernel_fun(Xtest, X, *args)
    
    u = np.linalg.solve(K + lambda_ * np.eye(len(y)), y)
    
    return np.sign(Ktest @ u)
