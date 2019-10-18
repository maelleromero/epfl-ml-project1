# -*- coding: utf-8 -*-
"""Solver"""

import numpy as np

def gradient_descent(function_object, w, max_evaluations, verbose=False, *args):
    """
    Find minimum

    Uses line search gradient descent to optimize function object

    :param function_object: loss function that returns loss and gradient
    :param w: weight
    :param max_evaluations: maximum number of evalutaions
    :param args: additional arguments (y, X)
    :param verbose: print output
    :return: weight, loss
    """
    # set an optimality stopping criterion
    linesearch_optTol = 1e-2

    # linesearch param
    linesearch_beta = 1e-4

    # evaluate the initial function value and gradient
    f, g = function_object(w, *args)
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
            f_new, g_new = function_object(w_new, *args)

            w_evals += 1
            evals += 1

            if f_new <= f - linesearch_beta * gamma * gg:
                # we have found a good enough gamma to decrease the loss function
                break

            if verbose:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # update step size
            if np.isinf(f_new):
                gamma = 1. / (10 ** w_evals)
            else:
                gamma = (gamma ** 2) * gg / (2. * (f_new - f + gamma * gg))

        # print progress
        if verbose:
            print("%d - loss: %.3f" % (evals, f_new))

        # update step-size for next iteration
        y = g_new - g
        gamma = -gamma * np.dot(y.T, g) / np.dot(y.T, y)

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

        if evals >= max_evaluations:
            if verbose:
                print("Reached maximum number of function evaluations %d" % max_evaluations)
            break

    return w, f

def gradient_descent_L1(function_object, w, L1_lammy, max_evaluations, *args, verbose=False):
    """
    Find minimum L1

    Uses the L1 proximal gradient descent to optimize loss
    The algorithm divides the step size by 2 until the right step
    size is found (decrease of L1 regularized loss function)

    :param function_object: loss function
    :param w: weight
    :param L1_lammy: lambda
    :param max_evaluations: maximum number of evaluations
    :param args: additional arguments (y, X)
    :param verbose: print output
    :return: weight, loss
    """

    # parameters of the optimization
    linesearch_optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = function_object(w, *args)
    evals = 1

    alpha = 1.
    proxL1 = lambda w, alpha: np.sign(w) * np.maximum(abs(w) - lambda_ * alpha, 0)
    L1Term = lambda w: L1_lammy * np.sum(np.abs(w))

    while True:
        gtd = None
        # start line search to determine alpha
        while True:
            w_new = w - alpha * g
            w_new = proxL1(w_new, alpha)

            if gtd is None:
                gtd = g.T.dot(w_new - w)

            f_new, g_new = function_object(w_new, *args)
            evals += 1

            if f_new + L1Term(w_new) <= f + L1Term(w) + gamma * alpha * gtd:
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
        alpha = -alpha * np.dot(y.T, g) / np.dot(y.T, y)

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

        if evals >= max_evaluations:
            if verbose:
                print("Reached maximum number of function evaluations %d" % max_evaluations)
            break

    return w, f