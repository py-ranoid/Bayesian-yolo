#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compression Tools


Karen Ullrich, Oct 2017

References:

    [1] Michael T. Heath. 1996. Scientific Computing: An Introductory Survey (2nd ed.). Eric M. Munson (Ed.). McGraw-Hill Higher Education. Chapter 1
"""

import numpy as np

# -------------------------------------------------------
# General tools
# -------------------------------------------------------


def unit_round_off(t=23):
    """
    :param t:
        number significand bits
    :return:
        unit round off based on nearest interpolation, for reference see [1]
    """
    return 0.5 * 2. ** (1. - t)


SIGNIFICANT_BIT_PRECISION = [unit_round_off(t=i + 1) for i in range(23)]


def float_precision(x):

    out = np.sum([x < sbp for sbp in SIGNIFICANT_BIT_PRECISION])
    return out


def float_precisions(X, dist_fun, layer=1):
    X = X.flatten()
    out = [float_precision(2 * x) for x in X]
    out = np.ceil(dist_fun(out))
    return out


def special_round(input, significant_bit):
    delta = unit_round_off(t=significant_bit)
    rounded = np.floor(input / delta + 0.5)
    rounded = rounded * delta
    return rounded


def fast_infernce_weights(w, significant_bit):
    # return w
    return special_round(w, significant_bit)


prev_conv_shape = None


def compress_matrix(x, verbose):
    global prev_conv_shape
    if verbose:
        print(x.shape, end="\t-->")
    if len(x.shape) != 2:
        A, B, C, D = x.shape
        x = x.reshape(A * B,  C * D)
        # remove non-necessary filters and rows
        x = x[:, (x != 0).any(axis=0)]
        x = x[(x != 0).any(axis=1), :]
        if verbose:
            print (x.shape, end='\t--> ')
        if prev_conv_shape == None:
            x = x.reshape(-1, B,  C, D)
            prev_conv_shape = x.shape
        else:
            x = x.reshape(-1, prev_conv_shape[0],  C, D)
            prev_conv_shape = x.shape
    else:
        # remove unnecessary rows, columns
        x = x[(x != 0).any(axis=1), :]  # remove row that are completely 0
        x = x[:, (x != 0).any(axis=0)]  # remove col that are completely 0_com
    if verbose:
        print (x.shape)
    return x


def extract_pruned_params(layers, wt_masks, bs_masks):

    post_weight_mus = []
    post_weight_vars = []
    post_bias_mus = []
    for i, (layer, wmask, bmask) in enumerate(zip(layers, wt_masks, bs_masks)):
        # compute posteriors
        post_weight_mu, post_weight_var = layer.compute_posterior_params()
        post_weight_var = post_weight_var.cpu().data.numpy()
        post_weight_mu = post_weight_mu.cpu().data.numpy()
        # apply mask to mus and variances

        post_weight_mu = post_weight_mu * wmask
        post_weight_var = post_weight_var * wmask

        post_bias_mu = layer.bias_mu.cpu().data.numpy() * bmask

        post_weight_mus.append(post_weight_mu)
        post_weight_vars.append(post_weight_var)
        post_bias_mus.append(post_bias_mu)

    return post_weight_mus, post_weight_vars, post_bias_mus


# -------------------------------------------------------
#  Compression rates (fast inference scenario)
# -------------------------------------------------------


def _compute_compression_rate(vars, in_precision=32., dist_fun=lambda x: np.max(x), overflow=10e38, compress_verbose=False):
    # compute in  number of bits occupied by the original architecture
    sizes = [v.size for v in vars]
    num_weights = float(np.sum(sizes))
    IN_BITS = in_precision * num_weights
    # prune architecture
    post_vars = [compress_matrix(v, compress_verbose) for v in vars]
    post_sizes = [v.size for v in post_vars]
    post_num_weights = float(np.sum(post_sizes))
    # compute
    significant_bits = [float_precisions(
        v, dist_fun, layer=k + 1) for k, v in enumerate(post_vars)]
    exponent_bit = np.ceil(np.log2(np.log2(overflow) + 1.) + 1.)
    print ("SIGNIFICANT BITS:", significant_bits)
    print ("EXPONENT BITS:", exponent_bit)
    total_bits = [1. + exponent_bit + sb for sb in significant_bits]
    OUT_BITS = np.sum(np.asarray(post_sizes) * np.asarray(total_bits))
    return num_weights / post_num_weights, IN_BITS / OUT_BITS, significant_bits, exponent_bit


def compute_compression_rate(layers, masks):
    # reduce architecture
    weight_mus, weight_vars, bias_mus = extract_pruned_params(
        layers, wt_masks=masks[0], bs_masks=masks[1])
    # compute overflow level based on maximum weight
    highest_weights = [np.max(np.abs(w)) for w in weight_mus]
    overflow = np.max(highest_weights)
    # compute compression rate
    CR_architecture, CR_fast_inference, _, _ = _compute_compression_rate(
        weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow)
    print("Compressing the architecture will degrease the model by a factor of %.1f." % (
        CR_architecture))
    print("Making use of weight uncertainty can reduce the model by a factor of %.1f." % (
        CR_fast_inference))


def compute_reduced_weights(layers, masks, prune=False):
    print ("Computing reduced weights")
    global prev_conv_shape
    weight_mus, weight_vars, bias_mus = extract_pruned_params(
        layers, wt_masks=masks[0], bs_masks=masks[1])
    overflow = np.max([np.max(np.abs(w)) for w in weight_mus])
    prev_conv_shape = None
    _, _, significant_bits, exponent_bits = _compute_compression_rate(
        weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow)

    if prune:

        print ("Pruning weights now :")
        prev_conv_shape = None
        post_weights = [compress_matrix(v, True) for v in weight_mus]

        print ("Pruning biases now :")
        prev_conv_shape = None
        post_biases = [compress_matrix(v.reshape(-1, 1), True)
                       for v in bias_mus]
        trimmed_weights = [fast_infernce_weights(weight_mu, significant_bit) for weight_mu, significant_bit in
                           zip(post_weights, significant_bits)]

        # Rounding off biases with the same precision as the layer's weights
        trimmed_biases = [fast_infernce_weights(bias, significant_bit)
                          for bias, significant_bit in zip(post_biases, significant_bits)]
    else:
        trimmed_weights = [fast_infernce_weights(weight_mu, significant_bit) for weight_mu, significant_bit in
                           zip(weight_mus, significant_bits)]

        # Rounding off biases with the same precision as the layer's weights
        trimmed_biases = [fast_infernce_weights(bias, significant_bit)
                          for bias, significant_bit in zip(bias_mus, significant_bits)]

    # return post_weights, post_biases
    return trimmed_weights, trimmed_biases
