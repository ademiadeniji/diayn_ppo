"""Variational Autoencoder (VAE)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf

from tf_agents.networks import network


def tanh_squash_to_spec(inputs, spec):
    """Maps inputs to range defined by spec using `tanh`."""
    means = (spec.maximum + spec.minimum) / 2.0
    magnitudes = (spec.maximum - spec.minimum) / 2.0

    return means + magnitudes * tf.tanh(inputs)


@gin.configurable
class VaeEncoder(network.Network):
    """Creates a VAE encoder.
    """

    def __init__(self,
                 sample_spec,
                 context_spec=None,
                 dim_z=16,
                 fc_layer_params=(256, 256),
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 dtype=tf.float32,
                 name='VaeEncoder'):
        """Creates an instance of `VaeEncoder`.

        Args:
            sample_spec: A nest of `tensor_spec.TensorSpec`
                representing the data.
            context_spec: A nest of `tensor_spec.TensorSpec`
                representing the context.
            dim_z: Dimension of the latent code.
            fc_layer_params: Optional list of fully_connected parameters,
                where each item is the number of units in the layer.
            activation_fn: Activation function, e.g. tf.nn.relu,
                slim.leaky_relu, ...
            kernel_initializer: Initializer to use for the kernels of the conv
                and dense layers. If none is provided a default glorot_uniform.
            dtype: The dtype to use by the convolution and fully connected
                layers.
            name: A string representing name of the network.

        Raises:
            ValueError: If `input_tensor_spec` contains more than one
                observation.
        """
        if not kernel_initializer:
            kernel_initializer = (
                tf.compat.v1.keras.initializers.glorot_uniform())

        encoding_layers = []

        if fc_layer_params:
            for num_units in fc_layer_params:
                encoding_layers.append(
                    tf.keras.layers.Dense(
                        num_units,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        dtype=dtype,
                        name='%s/fc' % name)
                )

        projection_layer = tf.keras.layers.Dense(
            2 * dim_z, name='gaussian_params')

        if context_spec is None:
            input_tensor_spec = sample_spec, context_spec
            is_conditional = False
        else:
            input_tensor_spec = sample_spec
            is_conditional = True

        super(VaeEncoder, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._dim_z = dim_z
        self._is_conditional = is_conditional
        self._encoding_layers = encoding_layers
        self._projection_layer = projection_layer

    @property
    def dim_z(self):
        return self._dim_z

    def call(self, inputs, step_type=None, training=False):
        if self._is_conditional:
            xs, contexts = inputs
            states = tf.keras.layers.concatenate([xs, contexts], axis=-1)
        else:
            states = tf.keras.layers.concatenate(inputs, axis=-1)

        for layer in self._encoding_layers:
            states = layer(states, training=training)

        gaussian_params = self._projection_layer(states)

        z_means = tf.identity(
            gaussian_params[..., :self._dim_z], name='z_means')
        z_stddevs = tf.add(
            tf.nn.softplus(gaussian_params[..., self._dim_z:]), 1e-6,
            name='z_stddevs')

        return z_means, z_stddevs


@gin.configurable
class VaeDecoder(network.Network):
    """Creates a VAE encoder.
    """

    def __init__(self,
                 sample_spec,
                 context_spec=None,
                 dim_z=16,
                 fc_layer_params=(256, 256),
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 tanh_squash_to_spec=True,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='VaeEncoder'):
        """Creates an instance of `VaeEncoder`.

        Args:
            sample_spec: A nest of `tensor_spec.TensorSpec`
                representing the data.
            context_spec: A nest of `tensor_spec.TensorSpec`
                representing the context.
            dim_z: Dimension of the latent code.
            fc_layer_params: Optional list of fully_connected parameters,
                where each item is the number of units in the layer.
            activation_fn: Activation function, e.g. tf.nn.relu,
                slim.leaky_relu, ...
            kernel_initializer: Initializer to use for the kernels of the conv
                and dense layers. If none is provided a default glorot_uniform.
            batch_squash: If True the outer_ranks of the observation are
                squashed into the batch dimension. This allow encoding networks
                to be used with observations with shape [BxTx...].
            dtype: The dtype to use by the convolution and fully connected
                layers.
            name: A string representing name of the network.

        Raises:
            ValueError: If `input_tensor_spec` contains more than one
                observation.
        """
        if len(tf.nest.flatten(sample_spec)) != 1:
            raise ValueError('This network only supports single spec outputs.')

        z_spec = tf.TensorSpec([dim_z], name='z')

        if not kernel_initializer:
            kernel_initializer = (
                tf.compat.v1.keras.initializers.glorot_uniform())

        encoding_layers = []

        if fc_layer_params:
            for num_units in fc_layer_params:
                encoding_layers.append(
                    tf.keras.layers.Dense(
                        num_units,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        dtype=dtype,
                        name='%s/fc' % name)
                )

        projection_layer = tf.keras.layers.Dense(self._dim_sample)

        if context_spec is None:
            input_tensor_spec = z_spec, context_spec
            is_conditional = False
        else:
            input_tensor_spec = z_spec
            is_conditional = True

        super(VaeDecoder, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._sample_spec = sample_spec
        self._dim_sample = np.prod(tf.nest.flatten(sample_spec.shape))
        self._dim_z = dim_z
        self._tanh_squash_to_spec = tanh_squash_to_spec
        self._is_conditional = is_conditional

        self._encoding_layers = encoding_layers
        self._projection_layer = projection_layer

    @property
    def dim_z(self):
        return self._dim_z

    def call(self, inputs, step_type=None, training=False):
        if self._is_conditional:
            zs, contexts = inputs
            states = tf.keras.layers.concatenate([zs, contexts], axis=-1)
        else:
            states = tf.keras.layers.concatenate(inputs, axis=-1)

        for layer in self._encoding_layers:
            states = layer(states, training=training)

        outputs = self._projection_layer(states)

        if self._tanh_squash_to_spec:
            outputs = tanh_squash_to_spec(outputs, self._sample_spec)

        return outputs
