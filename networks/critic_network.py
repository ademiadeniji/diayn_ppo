r"""Critic Network.
"""

import gin
import tensorflow as tf
import pdb
from tf_agents.networks import network
from tf_agents.networks import utils


@gin.configurable
class CriticNetwork(network.Network):
    """Creates a critic network."""

    def __init__(self,
                 input_tensor_spec,
                 preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 observation_dropout_layer_params=None,
                 action_fc_layer_params=None,
                 action_dropout_layer_params=None,
                 joint_fc_layer_params=None,
                 joint_dropout_layer_params=None,
                 activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 mask_xy=False,
                 name='CriticNetwork'):
        
        super(CriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._mask_xy = mask_xy

        observation_spec, action_spec = input_tensor_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]

        self._observation_layers = utils.mlp_layers(
            observation_conv_layer_params,
            observation_fc_layer_params,
            observation_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='observation_encoding')

        self._action_layers = utils.mlp_layers(
            None,
            action_fc_layer_params,
            action_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='action_encoding')

        self._joint_layers = utils.mlp_layers(
            None,
            joint_fc_layer_params,
            joint_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='joint_mlp')

        self._joint_layers.append(
            tf.keras.layers.Dense(
                1,
                activation=output_activation_fn,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                        minval=-0.003, maxval=0.003),
                name='value'))

        self._preprocessing_combiner = preprocessing_combiner

    def call(self, inputs, step_type=(), network_state=(), training=False):
        observations, actions = inputs
        if self._mask_xy:
            observations["observation"] = observations["observation"][:, 2:]
        
        # observations = observations
        del step_type  # unused.

        if self._preprocessing_combiner is not None:
            observations = self._preprocessing_combiner(observations)

        observations = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
        for layer in self._observation_layers:
            observations = layer(observations, training=training)

        actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
        for layer in self._action_layers:
            actions = layer(actions, training=training)

        joint = tf.concat([observations, actions], 1)
        for layer in self._joint_layers:
            joint = layer(joint, training=training)

        return tf.reshape(joint, [-1]), network_state
