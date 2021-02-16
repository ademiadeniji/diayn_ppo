"""Actor Policy based on an actor network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import reparameterized_sampling
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from collections import OrderedDict
import pdb

@gin.configurable
class LatentActorPolicy(tf_policy.Base):
    """Class to build Actor Policies."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 actor_network,
                 generator_network,
                 info_spec=(),
                 observation_normalizer=None,
                 clip=True,
                 training=False,
                 name=None):
        """Builds an Actor Policy given a actor network.

        Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of BoundedTensorSpec representing the actions.
            actor_network: An instance of a
                `tf_agents.networks.network.Network` to be used by the policy.
                The network will be called with call(observation, step_type,
                policy_state) and should return (actions_or_distributions,
                new_state).
            info_spec: A nest of TensorSpec representing the policy info.
            observation_normalizer: An object to use for observation
                normalization.
            clip: Whether to clip actions to spec before returning them.
                Default True. Most policy-based algorithms (PCL, PPO,
                REINFORCE) use unclipped continuous actions for training.
            training: Whether the network should be called in training mode.
            name: The name of this policy. All variables in this module will
                fall under that name. Defaults to the class name.

        Raises:
            ValueError: if actor_network is not of type network.Network.
        """
        if not isinstance(actor_network, network.Network):
            raise ValueError('actor_network must be a network.Network. Found '
                             '{}.'.format(type(actor_network)))
        actor_network.create_variables()
        self._actor_network = actor_network
        self._generator_network = generator_network
        self._observation_normalizer = observation_normalizer
        self._training = training
        self._steps_per_option = 1
        self._reset()

        super(LatentActorPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=actor_network.state_spec,
            info_spec=info_spec,
            clip=clip,
            name=name)

    def _apply_actor_network(self, time_step, policy_state):
        observation = time_step.observation
        if self._observation_normalizer:
            observation = self._observation_normalizer.normalize(observation)
        return self._actor_network(
            observation, time_step.step_type, policy_state,
            training=self._training)

    @property
    def observation_normalizer(self):
        return self._observation_normalizer

    def _variables(self):
        return (self._actor_network.variables +
                self._generator_network.variables)

    def _distribution(self, time_step, policy_state, seed=1):
        seed_stream = tfp.util.SeedStream(seed=seed, salt='ppo_policy')
        distribution_step = self._latent_distribution(time_step, policy_state)
        latent_actions = tf.nest.map_structure(
            lambda d: reparameterized_sampling.sample(d, seed=seed_stream()),
            distribution_step.action)
        action_distribution, _ = self._generator_network(OrderedDict({"observation":
            time_step.observation, "z":latent_actions}), time_step.step_type, policy_state)
            
        def _to_distribution(action_or_distribution):
            if isinstance(action_or_distribution, tf.Tensor):
                # This is an action tensor, so wrap it in a deterministic
                # distribution.
                return tfp.distributions.Deterministic(
                    loc=action_or_distribution)
            return action_or_distribution

        distributions = tf.nest.map_structure(_to_distribution,
                                              action_distribution)
        return policy_step.PolicyStep(distributions, policy_state)

    def latent_distribution(self, time_step, policy_state=()):
        """Generates the distribution over next actions given the time_step.
        """
        tf.nest.assert_same_structure(time_step, self._time_step_spec)
        tf.nest.assert_same_structure(policy_state, self._policy_state_spec)
        if self._automatic_state_reset:
            policy_state = self._maybe_reset_state(time_step, policy_state)
        step = self._latent_distribution(time_step=time_step,
                                         policy_state=policy_state)
        if self.emit_log_probability:
            raise NotImplementedError
        tf.nest.assert_same_structure(step, self._policy_step_spec)
        return step

    def _latent_distribution(self, time_step, policy_state):
        # Actor network outputs nested structure of distributions or actions.
        actions_or_distributions, policy_state = self._apply_actor_network(
            time_step, policy_state)

        def _to_distribution(action_or_distribution):
            if isinstance(action_or_distribution, tf.Tensor):
                # This is an action tensor, so wrap it in a deterministic
                # distribution.
                return tfp.distributions.Deterministic(
                    loc=action_or_distribution)
            return action_or_distribution

        distributions = tf.nest.map_structure(_to_distribution,
                                              actions_or_distributions)
        return policy_step.PolicyStep(distributions, policy_state)

    def _reset(self):
        self._t = 0

    def _action(self, time_step, policy_state, seed=1):
        """Implementation of `action`.

        Args:
            time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
            policy_state: A Tensor, or a nested dict, list or tuple of Tensors
                representing the previous policy_state.
            seed: Seed to use if action performs sampling (optional).

        Returns:
            A `PolicyStep` named tuple containing:
                `action`: An action Tensor matching the `action_spec()`.
                `state`: A policy state tensor to be fed into the next call.
                `info`: Optional information such as action log probabilities.
        """
        if self._t % self._steps_per_option == 0:
            seed_stream = tfp.util.SeedStream(seed=seed, salt='ppo_policy')
            distribution_step = self._latent_distribution(time_step, policy_state)
            latent_actions = tf.nest.map_structure(
                lambda d: reparameterized_sampling.sample(d, seed=seed_stream()),
                distribution_step.action)
            # policy_state = (distribution_step, latent_actions)
        self._t += 1
        # (distribution_step, latent_actions) = policy_state
        action_distribution, _ = self._generator_network(OrderedDict({"observation":
            time_step.observation, "z": latent_actions}), time_step.step_type, policy_state)
        if self.emit_log_probability:
            raise NotImplementedError
        info = distribution_step.info
        actions = tf.nest.map_structure(
            lambda d: d.sample(), action_distribution)
        return distribution_step._replace(action=actions, info=info, state=policy_state)