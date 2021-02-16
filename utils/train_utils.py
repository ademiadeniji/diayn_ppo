from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import gin
import numpy as np
import tensorflow as tf

from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

class MaxReturnMetric(tf_metrics.AverageReturnMetric):
    """Metric to compute the maximum return."""

    def __init__(self,
                 name='MaxReturnMetric',
                 prefix='Metrics',
                 dtype=tf.float32,
                 batch_size=1,
                 buffer_size=10,
                 inf=1e9):
        super(MaxReturnMetric, self).__init__(
            name=name,
            prefix=prefix,
            dtype=dtype,
            batch_size=batch_size,
            buffer_size=buffer_size)
        self._inf = inf

    @common.function(autograph=True)
    def result(self):
        if tf.equal(self._buffer.length, 0):
            return self._inf * tf.ones((), dtype=self._dtype)
        return tf.math.reduce_max(self._buffer.data)

class MinReturnMetric(tf_metrics.AverageReturnMetric):
    """Metric to compute the minimum return."""

    def __init__(self,
                 name='MinReturnMetric',
                 prefix='Metrics',
                 dtype=tf.float32,
                 batch_size=1,
                 buffer_size=10,
                 inf=1e9):
        super(MinReturnMetric, self).__init__(
            name=name,
            prefix=prefix,
            dtype=dtype,
            batch_size=batch_size,
            buffer_size=buffer_size)
        self._inf = inf

    @common.function(autograph=True)
    def result(self):
        if tf.equal(self._buffer.length, 0):
            return - self._inf * tf.ones((), dtype=self._dtype)
        return tf.math.reduce_min(self._buffer.data)

