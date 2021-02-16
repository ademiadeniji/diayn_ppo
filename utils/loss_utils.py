"""Loss utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# tf = tf.compat.v1


EPS = 1e-20


@tf.function
def l2_loss(targets, outputs, weights=1.0):
    loss = 0.5 * tf.math.reduce_sum(tf.square(targets - outputs), axis=-1)
    return tf.compat.v1.losses.compute_weighted_loss(loss, weights)


@tf.function
def angle_loss(targets, outputs, weights=1.0):
    targets_shape = targets.get_shape()
    outputs_shape = outputs.get_shape()
    assert len(targets_shape) == len(outputs_shape), (
        'Shapes are %r and %r.' % (targets_shape, outputs_shape))

    num_dims = len(targets_shape)
    for i in range(num_dims):
        assert targets_shape[i] == outputs_shape[i], (
            'Shapes are %r and %r.' % (targets_shape, outputs_shape))

    cos_dist = (tf.math.reduce_sum(targets * outputs, axis=-1) /
                tf.norm(targets, axis=-1) /
                tf.norm(outputs, axis=-1))
    loss = 1.0 - cos_dist
    return tf.compat.v1.losses.compute_weighted_loss(loss, weights)


@tf.function
def entropy(prob, weights=1.0):
    assert prob.get_shape()[-1] > 0
    prob = tf.abs(prob) + EPS
    entropy = -tf.math.reduce_sum(prob * tf.math.log(prob), axis=-1)
    return tf.compat.v1.losses.compute_weighted_loss(entropy, weights)


@tf.function
def kl_divergence(p, q, weights=1.0):
    assert p.get_shape()[-1] > 0
    assert q.get_shape()[-1] > 0
    p = tf.abs(p) + EPS
    q = tf.abs(q) + EPS
    loss = tf.math.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=-1)
    return tf.compat.v1.losses.compute_weighted_loss(loss, weights)


@tf.function
def kl_divergence_gaussian(mean1, stddev1, mean2, stddev2, weights=1.0):
    loss = (
        - 0.5 * tf.ones_like(mean1)
        + tf.math.log(stddev2 + EPS)
        - tf.math.log(stddev1 + EPS)
        + 0.5 * tf.square(stddev1) / (tf.square(stddev2) + EPS)
        + 0.5 * tf.square(mean2 - mean1) / (tf.square(stddev2) + EPS)
    )
    loss = tf.math.reduce_sum(loss, axis=-1)
    return tf.compat.v1.losses.compute_weighted_loss(loss, weights)


@tf.function
def hellinger_distance(p, q, weights=1.0):
    loss = 1.0 - tf.math.reduce_sum(tf.sqrt(p) * tf.sqrt(q), axis=-1)
    return tf.compat.v1.losses.compute_weighted_loss(loss, weights)


@tf.function
def log_normal(x, mean, stddev):
    stddev = tf.abs(stddev)
    stddev = tf.add(stddev, EPS)

    return -0.5 * tf.math.reduce_sum(
      (tf.math.log(2 * np.pi) + tf.math.log(tf.square(stddev))
       + tf.square(x - mean) / tf.square(stddev)),
      axis=-1)


@tf.function
def normal_kld(z, z_mean, z_stddev, weights=1.0):
    kld_array = (log_normal(z, z_mean, z_stddev) -
                 log_normal(z, 0.0, 1.0))
    return tf.compat.v1.losses.compute_weighted_loss(kld_array, weights)
