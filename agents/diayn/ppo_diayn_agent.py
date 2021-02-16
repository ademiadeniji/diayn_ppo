# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A PPO Agent.

Implements the PPO algorithm from (Schulman, 2017):
https://arxiv.org/abs/1707.06347

PPO is a simplification of the TRPO algorithm, both of which add stability to
policy gradient RL, while allowing multiple updates per batch of on-policy data,
by limiting the KL divergence between the policy that sampled the data and the
updated policy.

TRPO enforces a hard optimization constraint, but is a complex algorithm, which
often makes it harder to use in practice. PPO approximates the effect of TRPO
by using a soft constraint. There are two methods presented in the paper for
implementing the soft constraint: an adaptive KL loss penalty, and
limiting the objective value based on a clipped version of the policy importance
ratio. This code implements both, and allows the user to use either method or
both by modifying hyperparameters.

The importance ratio clipping is described in eq (7) and the adaptive KL penatly
is described in eq (8) of https://arxiv.org/pdf/1707.06347.pdf
- To disable IR clipping, set the importance_ratio_clipping parameter to 0.0
- To disable the adaptive KL penalty, set the initial_adaptive_kl_beta parameter
  to 0.0
- To disable the fixed KL cutoff penalty, set the kl_cutoff_factor parameter
  to 0.0

In order to compute KL divergence, the replay buffer must store action
distribution parameters from data collection. For now, it is assumed that
continuous actions are represented by a Normal distribution with mean & stddev,
and discrete actions are represented by a Categorical distribution with logits.

Note that the objective function chooses the lower value of the clipped and
unclipped objectives. Thus, if the importance ratio exceeds the clipped bounds,
then the optimizer will still not be incentivized to pass the bounds, as it is
only optimizing the minimum.

Advantage is computed using Generalized Advantage Estimation (GAE):
https://arxiv.org/abs/1506.02438
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pdb
import numpy as np

from absl import logging

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import tensor_normalizer
from tf_agents.utils import value_ops

from policies import ppo_policy

PPOLossInfo = collections.namedtuple('PPOLossInfo', (
    'policy_gradient_loss',
    'value_estimation_loss',
    'l2_regularization_loss',
    'entropy_regularization_loss',
    'kl_penalty_loss',
    'l2_distance_loss',
))


def _normalize_advantages(advantages, axes=(0,), variance_epsilon=1e-8):
  adv_mean, adv_var = tf.nn.moments(x=advantages, axes=axes, keepdims=True)
  normalized_advantages = ((advantages - adv_mean) /
                           (tf.sqrt(adv_var) + variance_epsilon))
  return normalized_advantages


@gin.configurable
class PPODiaynAgent(tf_agent.TFAgent):
  """A PPO Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               z_spec,
               optimizer=None,
               actor_net=None,
               value_net=None,
               inference_net=None,
               kl_posteriors_penalty=None,
               mock_inference=None,
               mock_reward=None,
               infer_from_com=None,
               l2_distance=None,
               rl_steps=None,
               inference_steps=None,
               importance_ratio_clipping=0.0,
               lambda_value=0.95,
               discount_factor=0.99,
               entropy_regularization=0.0,
               policy_l2_reg=0.0,
               value_function_l2_reg=0.0,
               value_pred_loss_coef=0.5,
               inference_loss_coef=1.0,
               num_epochs=25,
               use_gae=False,
               use_td_lambda_return=False,
               normalize_rewards=True,
               reward_norm_clipping=10.0,
               normalize_observations=True,
               log_prob_clipping=0.0,
               kl_cutoff_factor=2.0,
               kl_cutoff_coef=1000.0,
               initial_adaptive_kl_beta=1.0,
               adaptive_kl_target=0.01,
               adaptive_kl_tolerance=0.3,
               gradient_clipping=None,
               check_numerics=True,
               debug_summaries=None,
               summarize_grads_and_vars=None,
               train_step_counter=None,
               name=None):
    """Creates a PPO Diayn Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      optimizer: Optimizer to use for the agent.
      actor_net: A function actor_net(observations, action_spec) that returns
        tensor of action distribution params for each observation. Takes nested
        observation and returns nested action.
      value_net: A function value_net(time_steps) that returns value tensor from
        neural net predictions for each observation. Takes nested observation
        and returns batch of value_preds.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      lambda_value: Lambda parameter for TD-lambda computation.
      discount_factor: Discount factor for return computation.
      entropy_regularization: Coefficient for entropy regularization loss term.
      policy_l2_reg: Coefficient for l2 regularization of policy weights.
      value_function_l2_reg: Coefficient for l2 regularization of value function
        weights.
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss.
      num_epochs: Number of epochs for computing policy updates.
      use_gae: If True (default False), uses generalized advantage estimation
        for computing per-timestep advantage. Else, just subtracts value
        predictions from empirical return.
      use_td_lambda_return: If True (default False), uses td_lambda_return for
        training value function. (td_lambda_return = gae_advantage +
        value_predictions)
      normalize_rewards: If true, keeps moving variance of rewards and
        normalizes incoming rewards.
      reward_norm_clipping: Value above and below to clip normalized reward.
      normalize_observations: If true, keeps moving mean and variance of
        observations and normalizes incoming observations.
      log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
        values.  Default: no clipping.
      kl_cutoff_factor: If policy KL changes more than this much for any single
        timestep, adds a squared KL penalty to loss function.
      kl_cutoff_coef: Loss coefficient for kl cutoff term.
      initial_adaptive_kl_beta: Initial value for beta coefficient of adaptive
        kl penalty.
      adaptive_kl_target: Desired kl target for policy updates. If actual kl is
        far from this target, adaptive_kl_beta will be updated.
      adaptive_kl_tolerance: A tolerance for adaptive_kl_beta. Mean KL above (1
        + tol) * adaptive_kl_target, or below (1 - tol) * adaptive_kl_target,
        will cause adaptive_kl_beta to be updated.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      check_numerics: If true, adds tf.debugging.check_numerics to help find NaN
        / Inf values. For debugging only.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the actor_net is not a DistributionNetwork.
    """
    if not isinstance(actor_net, network.DistributionNetwork):
      raise ValueError(
          'actor_net must be an instance of a DistributionNetwork.')

    tf.Module.__init__(self, name=name)

    self._optimizer = optimizer
    self._actor_net = actor_net
    self._value_net = value_net
    self._inference_net = inference_net
    self._infer_from_com = infer_from_com
    self._kl_posteriors_penalty = kl_posteriors_penalty
    if self._kl_posteriors_penalty is not None:
      self._old_posteriors = None
    self._mock_inference = mock_inference
    self._mock_reward = mock_reward
    self._l2_distance = l2_distance
    self._rl_steps = rl_steps
    self._inference_steps = inference_steps
    self._importance_ratio_clipping = importance_ratio_clipping
    self._lambda = lambda_value
    self._discount_factor = discount_factor
    self._entropy_regularization = entropy_regularization
    self._policy_l2_reg = policy_l2_reg
    self._value_function_l2_reg = value_function_l2_reg
    self._value_pred_loss_coef = value_pred_loss_coef
    self._inference_loss_coef = inference_loss_coef
    self._num_epochs = num_epochs
    self._use_gae = use_gae
    self._use_td_lambda_return = use_td_lambda_return
    self._reward_norm_clipping = reward_norm_clipping
    self._log_prob_clipping = log_prob_clipping
    self._kl_cutoff_factor = kl_cutoff_factor
    self._kl_cutoff_coef = kl_cutoff_coef
    self._adaptive_kl_target = adaptive_kl_target
    self._adaptive_kl_tolerance = adaptive_kl_tolerance
    self._gradient_clipping = gradient_clipping or 0.0
    self._check_numerics = check_numerics

    self._z_spec = z_spec

    if tensor_spec.is_discrete(z_spec):
      n = int(z_spec.maximum) + 1
      logits = n * [1.0]
      self._z_prior = tfp.distributions.Categorical(logits=logits)
    else:
      self._z_prior = tfp.distributions.Uniform(
          low=z_spec.minimum, high=z_spec.maximum)

    if initial_adaptive_kl_beta > 0.0:
      # TODO(kbanoop): Rename create_variable.
      self._adaptive_kl_beta = common.create_variable(
          'adaptive_kl_beta', initial_adaptive_kl_beta, dtype=tf.float32)
    else:
      self._adaptive_kl_beta = None

    self._reward_normalizer = None
    if normalize_rewards:
      self._reward_normalizer = tensor_normalizer.StreamingTensorNormalizer(
          tensor_spec.TensorSpec([], tf.float32), scope='normalize_reward')
  
    self._observation_normalizer = None
    if normalize_observations:
      self._observation_normalizer = (
          tensor_normalizer.StreamingTensorNormalizer(
              time_step_spec.observation["observation"], scope='normalize_observations'))

    if self._rl_steps is not None and self._inference_steps is not None:
      self._rl_step_counter = tf.Variable(0, name="rl_step_counter")
      self._inference_step_counter = tf.Variable(0, name="inference_step_counter")
      self._training_rl = tf.Variable(True, name="training_rl")
      self._rl_steps = tf.constant(self._rl_steps, name="rl_steps")
      self._inference_steps = tf.constant(self._inference_steps, name="inference_steps")

    policy = greedy_policy.GreedyPolicy(
        ppo_policy.PPOPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=actor_net,
            value_network=value_net,
            observation_normalizer=self._observation_normalizer,
            clip=False,
            collect=False))

    collect_policy = ppo_policy.PPOPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        value_network=value_net,
        observation_normalizer=self._observation_normalizer,
        clip=False,
        collect=True)

    self._action_distribution_spec = (self._actor_net.output_spec)

    super(PPODiaynAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  @property
  def actor_net(self):
    """Returns actor_net TensorFlow template function."""
    return self._actor_net

  def _initialize(self):
    pass

  def compute_advantages(self, rewards, returns, discounts, value_preds):
    """Compute advantages, optionally using GAE.

    Based on baselines ppo1 implementation. Removes final timestep, as it needs
    to use this timestep for next-step value prediction for TD error
    computation.

    Args:
      rewards: Tensor of per-timestep rewards.
      returns: Tensor of per-timestep returns.
      discounts: Tensor of per-timestep discounts. Zero for terminal timesteps.
      value_preds: Cached value estimates from the data-collection policy.

    Returns:
      advantages: Tensor of length (len(rewards) - 1), because the final
        timestep is just used for next-step value prediction.
    """
    # Arg value_preds was appended with final next_step value. Make tensors
    #   next_value_preds by stripping first and last elements respectively.
    final_value_pred = value_preds[:, -1]
    value_preds = value_preds[:, :-1]

    if not self._use_gae:
      with tf.name_scope('empirical_advantage'):
        advantages = returns - value_preds
    else:
      advantages = value_ops.generalized_advantage_estimation(
          values=value_preds,
          final_value=final_value_pred,
          rewards=rewards,
          discounts=discounts,
          td_lambda=self._lambda,
          time_major=False)

    return advantages

  def get_epoch_loss(self, time_steps, next_time_steps, actions, act_log_probs, returns,
                     normalized_advantages, action_distribution_parameters,
                     weights, train_step, debug_summaries):
    """Compute the loss and create optimization op for one training epoch.

    All tensors should have a single batch dimension.

    Args:
      time_steps: A minibatch of TimeStep tuples.
      actions: A minibatch of actions.
      act_log_probs: A minibatch of action probabilities (probability under the
        sampling policy).
      returns: A minibatch of per-timestep returns.
      normalized_advantages: A minibatch of normalized per-timestep advantages.
      action_distribution_parameters: Parameters of data-collecting action
        distribution. Needed for KL computation.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      train_step: A train_step variable to increment for each train step.
        Typically the global_step.
      debug_summaries: True if debug summaries should be created.

    Returns:
      A tf_agent.LossInfo named tuple with the total_loss and all intermediate
        losses in the extra field contained in a PPOLossInfo named tuple.
    """
    # Evaluate the current policy on timesteps.

    # batch_size from time_steps
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._collect_policy.get_initial_state(batch_size)
    distribution_step = self._collect_policy.distribution(
        time_steps, policy_state)
    # TODO(eholly): Rename policy distributions to something clear and uniform.
    current_policy_distribution = distribution_step.action

    # Call all loss functions and add all loss values.
    value_estimation_loss = self.value_estimation_loss(time_steps, returns,
                                                       weights, debug_summaries)

    policy_gradient_loss = self.policy_gradient_loss(
            time_steps,
            actions,
            tf.stop_gradient(act_log_probs),
            tf.stop_gradient(normalized_advantages),
            current_policy_distribution,
            weights,
            debug_summaries=debug_summaries)
          
    if self._policy_l2_reg > 0.0 or self._value_function_l2_reg > 0.0:
      l2_regularization_loss = self.l2_regularization_loss(debug_summaries)
    else:
      l2_regularization_loss = tf.zeros_like(policy_gradient_loss)

    if self._entropy_regularization > 0.0:
      entropy_regularization_loss = self.entropy_regularization_loss(
          time_steps, current_policy_distribution, weights, debug_summaries)
    else:
      entropy_regularization_loss = tf.zeros_like(policy_gradient_loss)

    kl_penalty_loss = self.kl_penalty_loss(time_steps,
                                           action_distribution_parameters,
                                           current_policy_distribution, weights,
                                           debug_summaries)

    if self._l2_distance:
      l2_distance_loss = tf.cast(self.l2_distance_loss(time_steps, next_time_steps), dtype=tf.float32)
    else:
      l2_distance_loss = tf.zeros_like(policy_gradient_loss)
    
    total_loss = (policy_gradient_loss + value_estimation_loss + l2_regularization_loss + \
      entropy_regularization_loss + kl_penalty_loss + l2_distance_loss)

    return tf_agent.LossInfo(
        total_loss,
        PPOLossInfo(
            policy_gradient_loss=policy_gradient_loss,
            value_estimation_loss=value_estimation_loss,
            l2_regularization_loss=l2_regularization_loss,
            entropy_regularization_loss=entropy_regularization_loss,
            kl_penalty_loss=kl_penalty_loss,
            l2_distance_loss=l2_distance_loss,
        ))

  def intrinsic_reward(self, time_steps, next_time_steps):
    with tf.name_scope('intrinsic_reward'):
      zs = time_steps.observation['z']
      next_observations = next_time_steps.observation['observation']
      inference_input = time_steps.observation['observation'] - \
          next_time_steps.observation['observation']
      step_type = next_time_steps.step_type
      if self._infer_from_com:
        inference_input = inference_input[:, :, :2]
      if self._mock_inference:
        z_posteriors = self._cosine_similarity(inference_input, zs)
        log_z_posteriors = tf.math.log(z_posteriors)
      elif self._mock_reward:
        rewards = self._handcraft_rewards(inference_input, zs)
        return rewards
      else:
        z_posteriors, _ = self._inference_net(
            inference_input, step_type, ())
        log_z_posteriors = common.log_probability(
            z_posteriors, zs, self._z_spec)
      log_z_priors = common.log_probability(
          self._z_prior, zs, self._z_spec)
      rewards = log_z_posteriors - log_z_priors

      if self._debug_summaries:
        common.generate_tensor_summaries(
            'zs',
            zs,
            self.train_step_counter)
        common.generate_tensor_summaries(
            'log_z_posteriors',
            log_z_posteriors,
            step=self.train_step_counter)
        common.generate_tensor_summaries(
            'log_z_priors',
            log_z_priors,
            step=self.train_step_counter)
        common.generate_tensor_summaries(
            'rewards',
            rewards,
            step=self.train_step_counter)
    return rewards

  def compute_return_and_advantage(self, time_steps, next_time_steps, value_preds):
    """Compute the Monte Carlo return and advantage.

    Normalazation will be applied to the computed returns and advantages if
    it's enabled.

    Args:
      next_time_steps: batched tensor of TimeStep tuples after action is taken.
      value_preds: Batched value prediction tensor. Should have one more entry
        in time index than time_steps, with the final value corresponding to the
        value prediction of the final state.

    Returns:
      tuple of (return, normalized_advantage), both are batched tensors.
    """
    discounts = next_time_steps.discount * tf.constant(
        self._discount_factor, dtype=tf.float32)

    rewards = self.intrinsic_reward(time_steps, next_time_steps)
    if self._debug_summaries:
      # Summarize rewards before they get normalized below.
      tf.compat.v2.summary.histogram(
          name='rewards', data=rewards, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='rewards_mean',
          data=tf.reduce_mean(rewards),
          step=self.train_step_counter)

    # Normalize rewards if self._reward_normalizer is defined.
    if self._reward_normalizer:
      rewards = self._reward_normalizer.normalize(
          rewards, center_mean=False, clip_value=self._reward_norm_clipping)
      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
            name='rewards_normalized',
            data=rewards,
            step=self.train_step_counter)
        tf.compat.v2.summary.scalar(
            name='rewards_normalized_mean',
            data=tf.reduce_mean(rewards),
            step=self.train_step_counter)

    # Make discount 0.0 at end of each episode to restart cumulative sum
    #   end of each episode.
    episode_mask = common.get_episode_mask(next_time_steps)
    discounts *= episode_mask

    # Compute Monte Carlo returns.
    returns = value_ops.discounted_return(rewards, discounts, time_major=False)
    if self._debug_summaries:
      tf.compat.v2.summary.histogram(
          name='returns', data=returns, step=self.train_step_counter)

    # Compute advantages.
    advantages = self.compute_advantages(rewards, returns, discounts,
                                         value_preds)
    normalized_advantages = _normalize_advantages(advantages, axes=(0, 1))
    if self._debug_summaries:
      tf.compat.v2.summary.histogram(
          name='advantages', data=advantages, step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='advantages_normalized',
          data=normalized_advantages,
          step=self.train_step_counter)

    # Return TD-Lambda returns if both use_td_lambda_return and use_gae.
    if self._use_td_lambda_return:
      if not self._use_gae:
        logging.warning('use_td_lambda_return was True, but use_gae was '
                        'False. Using Monte Carlo return.')
      else:
        returns = tf.add(
            advantages, value_preds[:, :-1], name='td_lambda_returns')

    return returns, normalized_advantages

  def _reset_counter(self, step_counter, training_switch):
        step_counter.assign(0)
        training_switch.assign(~training_switch)
        return step_counter

  def _update_rl(self, rl_loss, rl_tape):
      variables_to_train = (
              self._actor_net.trainable_weights +
              self._value_net.trainable_weights) 
      grads = rl_tape.gradient(rl_loss, variables_to_train)
      grads_and_vars = tuple(zip(grads, variables_to_train))
      if self._gradient_clipping > 0:
        grads_and_vars = eager_utils.clip_gradient_norms(
            grads_and_vars, self._gradient_clipping)

      if self._summarize_grads_and_vars and self._debug_summaries:
        eager_utils.add_gradients_summaries(grads_and_vars,
                                            self.train_step_counter)
        eager_utils.add_variables_summaries(grads_and_vars,
                                            self.train_step_counter)

      self._optimizer.apply_gradients(
          grads_and_vars, global_step=self.train_step_counter)
      
      if self._inference_steps is not None and self._rl_steps is not None:
          self._rl_step_counter.assign_add(1)
          tf.cond(self._rl_step_counter == self._rl_steps, 
          lambda: self._reset_counter(self._rl_step_counter, self._training_rl), 
          lambda: self._rl_step_counter)
      return rl_loss
  
  def _update_inference(self, inference_loss, inference_tape):
      if not self._mock_inference and not self._mock_reward:
          inference_grads = inference_tape.gradient(inference_loss,
                                              self._inference_net.trainable_weights)
          grads_and_vars = tuple(zip(inference_grads, self._inference_net.trainable_weights))
          if self._gradient_clipping > 0:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)

          if self._summarize_grads_and_vars and self._debug_summaries:
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)

          self._optimizer.apply_gradients(
              grads_and_vars, global_step=self.train_step_counter)

      if self._inference_steps is not None and self._rl_steps is not None:
          self._inference_step_counter.assign_add(1)
          tf.cond(self._inference_step_counter == self._inference_steps, 
          lambda: self._reset_counter(self._inference_step_counter, self._training_rl), 
          lambda: self._inference_step_counter)
      return inference_loss

  def _train(self, experience, weights):
    # Get individual tensors from transitions.
    (time_steps, policy_steps_,
     next_time_steps) = trajectory.to_transition(experience)
    actions = policy_steps_.action

    if self._debug_summaries:
      actions_list = tf.nest.flatten(actions)
      show_action_index = len(actions_list) != 1
      for i, single_action in enumerate(actions_list):
        action_name = ('actions_{}'.format(i)
                       if show_action_index else 'actions')
        tf.compat.v2.summary.histogram(
            name=action_name, data=single_action, step=self.train_step_counter)

    action_distribution_parameters = policy_steps_.info

    # Reconstruct per-timestep policy distribution from stored distribution
    #   parameters.
    old_actions_distribution = (
        distribution_spec.nested_distributions_from_specs(
            self._action_distribution_spec, action_distribution_parameters))

    # Compute log probability of actions taken during data collection, using the
    #   collect policy distribution.
    act_log_probs = common.log_probability(old_actions_distribution, actions,
                                           self._action_spec)

    # Compute the value predictions for states using the current value function.
    # To be used for return & advantage computation.
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    value_state = self._collect_policy.get_initial_value_state(batch_size)

    value_preds, _ = self._collect_policy.apply_value_network(
        experience.observation, experience.step_type, value_state=value_state)
    value_preds = tf.stop_gradient(value_preds)

    valid_mask = ppo_utils.make_timestep_mask(next_time_steps)

    if weights is None:
      weights = valid_mask
    else:
      weights *= valid_mask

    returns, normalized_advantages = self.compute_return_and_advantage(
        time_steps, next_time_steps, value_preds)

    # Loss tensors across batches will be aggregated for summaries.
    policy_gradient_losses = []
    value_estimation_losses = []
    l2_regularization_losses = []
    entropy_regularization_losses = []
    kl_penalty_losses = []
    l2_distance_losses = []
    inference_losses = []

    loss_info = None  # TODO(b/123627451): Remove.
    # For each epoch, create its own train op that depends on the previous one.
    for i_epoch in range(self._num_epochs):
      with tf.name_scope('epoch_%d' % i_epoch):
        # Only save debug summaries for first and last epochs.
        debug_summaries = (
            self._debug_summaries and
            (i_epoch == 0 or i_epoch == self._num_epochs - 1))
        
        # RL and Inference Training Loop
        # Build one epoch train op.
        with tf.GradientTape() as rl_tape:
          loss_info = self.get_epoch_loss(time_steps, next_time_steps, actions, act_log_probs,
                                          returns, normalized_advantages,
                                          action_distribution_parameters,
                                          weights, self.train_step_counter,
                                          debug_summaries)

        trainable_inference_variables = self._inference_net.trainable_weights
        with tf.GradientTape(watch_accessed_variables=False) as inference_tape:
          assert trainable_inference_variables, (
              'No trainable inference variables to optimize.')
          inference_tape.watch(trainable_inference_variables)
          inference_loss = self.inference_loss(time_steps, actions, next_time_steps, weights)
          tf.debugging.check_numerics(inference_loss, 'inference loss is inf or nan.')
        
        if self._rl_steps is not None and self._inference_steps is not None:
            tf.cond(self._training_rl,  
            lambda: self._update_rl(loss_info.loss, rl_tape),
            lambda: self._update_inference(inference_loss, inference_tape))
        else:
            self._update_rl(loss_info.loss, rl_tape)
            self._update_inference(inference_loss, inference_tape)			

        policy_gradient_losses.append(loss_info.extra.policy_gradient_loss)
        value_estimation_losses.append(loss_info.extra.value_estimation_loss)
        l2_regularization_losses.append(loss_info.extra.l2_regularization_loss)
        entropy_regularization_losses.append(loss_info.extra.entropy_regularization_loss)
        kl_penalty_losses.append(loss_info.extra.kl_penalty_loss)
        l2_distance_losses.append(loss_info.extra.l2_distance_loss)
        inference_losses.append(inference_loss)

    # After update epochs, update adaptive kl beta, then update observation
    #   normalizer and reward normalizer.
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._collect_policy.get_initial_state(batch_size)
    # Compute the mean kl from previous action distribution.
    kl_divergence = self._kl_divergence(
        time_steps, action_distribution_parameters,
        self._collect_policy.distribution(time_steps, policy_state).action)
    self.update_adaptive_kl_beta(kl_divergence)

    if self._observation_normalizer:
      self._observation_normalizer.update(
          time_steps.observation["observation"], outer_dims=[0, 1])
    else:
      # TODO(b/127661780): Verify performance of reward_normalizer when obs are
      #                    not normalized
      if self._reward_normalizer:
        self._reward_normalizer.update(
            next_time_steps.reward, outer_dims=[0, 1])

    loss_info = tf.nest.map_structure(tf.identity, loss_info)

    # Make summaries for total loss averaged across all epochs.
    # The *_losses lists will have been populated by
    #   calls to self.get_epoch_loss. Assumes all the losses have same length.
    with tf.name_scope('Losses/'):
      num_epochs = len(policy_gradient_losses)
      total_policy_gradient_loss = tf.add_n(policy_gradient_losses) / num_epochs
      total_value_estimation_loss = tf.add_n(
          value_estimation_losses) / num_epochs
      total_l2_regularization_loss = tf.add_n(
          l2_regularization_losses) / num_epochs
      total_entropy_regularization_loss = tf.add_n(
          entropy_regularization_losses) / num_epochs
      total_kl_penalty_loss = tf.add_n(kl_penalty_losses) / num_epochs
      if not self._mock_reward:
        total_inference_loss = tf.add_n(inference_losses) / num_epochs
      else:
        total_inference_loss = 0.0
      if self._l2_distance:
        total_l2_distance_loss = tf.add_n(l2_distance_losses) / num_epochs
      else:
        total_l2_distance_loss = 0.0

      tf.compat.v2.summary.scalar(
          name='policy_gradient_loss',
          data=total_policy_gradient_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_estimation_loss',
          data=total_value_estimation_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='l2_regularization_loss',
          data=total_l2_regularization_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='entropy_regularization_loss',
          data=total_entropy_regularization_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='kl_penalty_loss',
          data=total_kl_penalty_loss,
          step=self.train_step_counter)
      if not self._mock_reward:
        tf.compat.v2.summary.scalar(
            name='inference_loss',
            data=total_inference_loss,
            step=self.train_step_counter)
      if self._l2_distance:
        tf.compat.v2.summary.scalar(
            name='l2_distance_loss',
            data=total_l2_distance_loss,
            step=self.train_step_counter)
    with tf.name_scope('MDP/'):
      common.generate_tensor_summaries(
            'observation',
            time_steps.observation['observation'],
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'next_observation',
            next_time_steps.observation['observation'],
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'actions',
            actions,
            step=self.train_step_counter)
      
      total_abs_loss = (
          tf.abs(total_policy_gradient_loss) +
          tf.abs(total_value_estimation_loss) +
          tf.abs(total_entropy_regularization_loss) +
          tf.abs(total_l2_regularization_loss) + tf.abs(total_kl_penalty_loss) +
          tf.abs(total_inference_loss) + tf.abs(total_l2_distance_loss))

      tf.compat.v2.summary.scalar(
          name='total_abs_loss',
          data=total_abs_loss,
          step=self.train_step_counter)

    if self._summarize_grads_and_vars:
      with tf.name_scope('Variables/'):
        if self._mock_inference or self._mock_reward:
          all_vars = (
              self._actor_net.trainable_weights +
              self._value_net.trainable_weights)
        else:
          all_vars = (
              self._actor_net.trainable_weights +
              self._value_net.trainable_weights +
              self._inference_net.trainable_weights)
        for var in all_vars:
          tf.compat.v2.summary.histogram(
              name=var.name.replace(':', '_'),
              data=var,
              step=self.train_step_counter)

    return loss_info
  
  def l2_distance_loss(self, time_steps, next_time_steps):
    observation = time_steps.observation['observation']
    next_observation = next_time_steps.observation['observation']
    if self._infer_from_com:
        observation = observation[:, :, :2]
        next_observation = next_observation[:, :, :2]
    return -(tf.norm(observation - next_observation)**2)

  def l2_regularization_loss(self, debug_summaries=False):
    if self._policy_l2_reg > 0 or self._value_function_l2_reg > 0:
      with tf.name_scope('l2_regularization'):
        # Regularize policy weights.
        policy_vars_to_l2_regularize = [
            v for v in self._actor_net.trainable_weights if 'kernel' in v.name
        ]
        policy_l2_losses = [
            tf.reduce_sum(input_tensor=tf.square(v)) * self._policy_l2_reg
            for v in policy_vars_to_l2_regularize
        ]

        # Regularize value function weights.
        vf_vars_to_l2_regularize = [
            v for v in self._value_net.trainable_weights if 'kernel' in v.name
        ]
        vf_l2_losses = [
            tf.reduce_sum(input_tensor=tf.square(v)) *
            self._value_function_l2_reg for v in vf_vars_to_l2_regularize
        ]

        l2_losses = policy_l2_losses + vf_l2_losses
        total_l2_loss = tf.add_n(l2_losses, name='l2_loss')

        if self._check_numerics:
          total_l2_loss = tf.debugging.check_numerics(total_l2_loss,
                                                      'total_l2_loss')

        if debug_summaries:
          tf.compat.v2.summary.histogram(
              name='l2_loss', data=total_l2_loss, step=self.train_step_counter)
    else:
      total_l2_loss = tf.constant(0.0, dtype=tf.float32, name='zero_l2_loss')

    return total_l2_loss

  def entropy_regularization_loss(self,
                                  time_steps,
                                  current_policy_distribution,
                                  weights,
                                  debug_summaries=False):
    """Create regularization loss tensor based on agent parameters."""
    if self._entropy_regularization > 0:
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)
      with tf.name_scope('entropy_regularization'):
        entropy = tf.cast(
            common.entropy(current_policy_distribution, self.action_spec),
            tf.float32)
        entropy_reg_loss = (
            tf.reduce_mean(input_tensor=-entropy * weights) *
            self._entropy_regularization)
        if self._check_numerics:
          entropy_reg_loss = tf.debugging.check_numerics(
              entropy_reg_loss, 'entropy_reg_loss')

        if debug_summaries:
          tf.compat.v2.summary.histogram(
              name='entropy_reg_loss',
              data=entropy_reg_loss,
              step=self.train_step_counter)
    else:
      entropy_reg_loss = tf.constant(
          0.0, dtype=tf.float32, name='zero_entropy_reg_loss')

    return entropy_reg_loss

  def value_estimation_loss(self,
                            time_steps,
                            returns,
                            weights,
                            debug_summaries=False):
    """Computes the value estimation loss for actor-critic training.

    All tensors should have a single batch dimension.

    Args:
      time_steps: A batch of timesteps.
      returns: Per-timestep returns for value function to predict. (Should come
        from TD-lambda computation.)
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.

    Returns:
      value_estimation_loss: A scalar value_estimation_loss loss.
    """
    observation = time_steps.observation
    if debug_summaries:
      observation_list = tf.nest.flatten(observation)
      show_observation_index = len(observation_list) != 1
      for i, single_observation in enumerate(observation_list):
        observation_name = ('observations_{}'.format(i)
                            if show_observation_index else 'observations')
        tf.compat.v2.summary.histogram(
            name=observation_name,
            data=single_observation,
            step=self.train_step_counter)

    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    value_state = self._collect_policy.get_initial_value_state(batch_size)

    value_preds, _ = self._collect_policy.apply_value_network(
        time_steps.observation, time_steps.step_type, value_state=value_state)
    value_estimation_error = tf.math.squared_difference(returns, value_preds)
    value_estimation_error *= weights

    value_estimation_loss = (
        tf.reduce_mean(input_tensor=value_estimation_error) *
        self._value_pred_loss_coef)
    if debug_summaries:
      tf.compat.v2.summary.scalar(
          name='value_pred_avg',
          data=tf.reduce_mean(input_tensor=value_preds),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_actual_avg',
          data=tf.reduce_mean(input_tensor=returns),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_estimation_loss',
          data=value_estimation_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='value_preds', data=value_preds, step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='value_estimation_error',
          data=value_estimation_error,
          step=self.train_step_counter)

    if self._check_numerics:
      value_estimation_loss = tf.debugging.check_numerics(
          value_estimation_loss, 'value_estimation_loss')

    return value_estimation_loss

  def policy_gradient_loss(self,
                           time_steps,
                           actions,
                           sample_action_log_probs,
                           advantages,
                           current_policy_distribution,
                           weights,
                           debug_summaries=False):
    """Create tensor for policy gradient loss.

    All tensors should have a single batch dimension.

    Args:
      time_steps: TimeSteps with observations for each timestep.
      actions: Tensor of actions for timesteps, aligned on index.
      sample_action_log_probs: Tensor of sample probability of each action.
      advantages: Tensor of advantage estimate for each timestep, aligned on
        index. Works better when advantage estimates are normalized.
      current_policy_distribution: The policy distribution, evaluated on all
        time_steps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.

    Returns:
      policy_gradient_loss: A tensor that will contain policy gradient loss for
        the on-policy experience.
    """
    tf.nest.assert_same_structure(time_steps, self.time_step_spec)
    action_log_prob = common.log_probability(current_policy_distribution,
                                             actions, self._action_spec)
    action_log_prob = tf.cast(action_log_prob, tf.float32)
    if self._log_prob_clipping > 0.0:
      action_log_prob = tf.clip_by_value(action_log_prob,
                                         -self._log_prob_clipping,
                                         self._log_prob_clipping)
    if self._check_numerics:
      action_log_prob = tf.debugging.check_numerics(action_log_prob,
                                                    'action_log_prob')

    # Prepare both clipped and unclipped importance ratios.
    importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)
    importance_ratio_clipped = tf.clip_by_value(
        importance_ratio, 1 - self._importance_ratio_clipping,
        1 + self._importance_ratio_clipping)

    if self._check_numerics:
      importance_ratio = tf.debugging.check_numerics(importance_ratio,
                                                     'importance_ratio')
      if self._importance_ratio_clipping > 0.0:
        importance_ratio_clipped = tf.debugging.check_numerics(
            importance_ratio_clipped, 'importance_ratio_clipped')

    # Pessimistically choose the minimum objective value for clipped and
    #   unclipped importance ratios.
    per_timestep_objective = importance_ratio * advantages
    per_timestep_objective_clipped = importance_ratio_clipped * advantages
    per_timestep_objective_min = tf.minimum(per_timestep_objective,
                                            per_timestep_objective_clipped)

    if self._importance_ratio_clipping > 0.0:
      policy_gradient_loss = -per_timestep_objective_min
    else:
      policy_gradient_loss = -per_timestep_objective

    policy_gradient_loss = tf.reduce_mean(input_tensor=policy_gradient_loss *
                                          weights)

    if debug_summaries:
      if self._importance_ratio_clipping > 0.0:
        clip_fraction = tf.reduce_mean(
            input_tensor=tf.cast(
                tf.greater(
                    tf.abs(importance_ratio -
                           1.0), self._importance_ratio_clipping), tf.float32))
        tf.compat.v2.summary.scalar(
            name='clip_fraction',
            data=clip_fraction,
            step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='action_log_prob',
          data=action_log_prob,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='action_log_prob_sample',
          data=sample_action_log_probs,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='importance_ratio',
          data=importance_ratio,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='importance_ratio_mean',
          data=tf.reduce_mean(input_tensor=importance_ratio),
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='importance_ratio_clipped',
          data=importance_ratio_clipped,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='per_timestep_objective',
          data=per_timestep_objective,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='per_timestep_objective_clipped',
          data=per_timestep_objective_clipped,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='per_timestep_objective_min',
          data=per_timestep_objective_min,
          step=self.train_step_counter)
      entropy = common.entropy(current_policy_distribution, self.action_spec)
      tf.compat.v2.summary.histogram(
          name='policy_entropy', data=entropy, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='policy_entropy_mean',
          data=tf.reduce_mean(input_tensor=entropy),
          step=self.train_step_counter)
      for i, (single_action, single_distribution) in enumerate(
          zip(
              tf.nest.flatten(self.action_spec),
              tf.nest.flatten(current_policy_distribution))):
        # Categorical distribution (used for discrete actions) doesn't have a
        # mean.
        distribution_index = '_{}'.format(i) if i > 0 else ''
        if not tensor_spec.is_discrete(single_action):
          tf.compat.v2.summary.histogram(
              name='actions_distribution_mean' + distribution_index,
              data=single_distribution.mean(),
              step=self.train_step_counter)
          tf.compat.v2.summary.histogram(
              name='actions_distribution_stddev' + distribution_index,
              data=single_distribution.stddev(),
              step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='policy_gradient_loss',
          data=policy_gradient_loss,
          step=self.train_step_counter)

    if self._check_numerics:
      policy_gradient_loss = tf.debugging.check_numerics(
          policy_gradient_loss, 'policy_gradient_loss')

    return policy_gradient_loss

  def kl_cutoff_loss(self, kl_divergence, debug_summaries=False):
    # Squared penalization for mean KL divergence above some threshold.
    if self._kl_cutoff_factor <= 0.0:
      return tf.constant(0.0, dtype=tf.float32, name='zero_kl_cutoff_loss')
    kl_cutoff = self._kl_cutoff_factor * self._adaptive_kl_target
    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)
    kl_over_cutoff = tf.maximum(mean_kl - kl_cutoff, 0.0)
    kl_cutoff_loss = self._kl_cutoff_coef * tf.square(kl_over_cutoff)

    if debug_summaries:
      tf.compat.v2.summary.scalar(
          name='kl_cutoff_count',
          data=tf.reduce_sum(
              input_tensor=tf.cast(kl_divergence > kl_cutoff, dtype=tf.int64)),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='kl_cutoff_loss',
          data=kl_cutoff_loss,
          step=self.train_step_counter)

    return tf.identity(kl_cutoff_loss, name='kl_cutoff_loss')

  def adaptive_kl_loss(self, kl_divergence, debug_summaries=False):
    if self._adaptive_kl_beta is None:
      return tf.constant(0.0, dtype=tf.float32, name='zero_adaptive_kl_loss')

    # Define the loss computation, which depends on the update computation.
    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)
    adaptive_kl_loss = self._adaptive_kl_beta * mean_kl

    if debug_summaries:
      tf.compat.v2.summary.scalar(
          name='adaptive_kl_loss',
          data=adaptive_kl_loss,
          step=self.train_step_counter)

    return adaptive_kl_loss

  def _kl_divergence(self, time_steps, action_distribution_parameters,
                     current_policy_distribution):
    outer_dims = list(
        range(nest_utils.get_outer_rank(time_steps, self.time_step_spec)))

    old_actions_distribution = (
        distribution_spec.nested_distributions_from_specs(
            self._action_distribution_spec, action_distribution_parameters))

    kl_divergence = ppo_utils.nested_kl_divergence(
        old_actions_distribution,
        current_policy_distribution,
        outer_dims=outer_dims)
    return kl_divergence

  def kl_penalty_loss(self,
                      time_steps,
                      action_distribution_parameters,
                      current_policy_distribution,
                      weights,
                      debug_summaries=False):
    """Compute a loss that penalizes policy steps with high KL.

    Based on KL divergence from old (data-collection) policy to new (updated)
    policy.

    All tensors should have a single batch dimension.

    Args:
      time_steps: TimeStep tuples with observations for each timestep. Used for
        computing new action distributions.
      action_distribution_parameters: Action distribution params of the data
        collection policy, used for reconstruction old action distributions.
      current_policy_distribution: The policy distribution, evaluated on all
        time_steps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Inlcudes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.

    Returns:
      kl_penalty_loss: The sum of a squared penalty for KL over a constant
        threshold, plus an adaptive penalty that encourages updates toward a
        target KL divergence.
    """
    kl_divergence = self._kl_divergence(time_steps,
                                        action_distribution_parameters,
                                        current_policy_distribution) * weights

    if debug_summaries:
      tf.compat.v2.summary.histogram(
          name='kl_divergence',
          data=kl_divergence,
          step=self.train_step_counter)

    kl_cutoff_loss = self.kl_cutoff_loss(kl_divergence, debug_summaries)
    adaptive_kl_loss = self.adaptive_kl_loss(kl_divergence, debug_summaries)
    return tf.add(kl_cutoff_loss, adaptive_kl_loss, name='kl_penalty_loss')

  def update_adaptive_kl_beta(self, kl_divergence):
    """Create update op for adaptive KL penalty coefficient.

    Args:
      kl_divergence: KL divergence of old policy to new policy for all
        timesteps.

    Returns:
      update_op: An op which runs the update for the adaptive kl penalty term.
    """
    if self._adaptive_kl_beta is None:
      return tf.no_op()

    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)

    # Update the adaptive kl beta after each time it is computed.
    mean_kl_below_bound = (
        mean_kl < self._adaptive_kl_target *
        (1.0 - self._adaptive_kl_tolerance))
    mean_kl_above_bound = (
        mean_kl > self._adaptive_kl_target *
        (1.0 + self._adaptive_kl_tolerance))
    adaptive_kl_update_factor = tf.case(
        [
            (mean_kl_below_bound,
             lambda: tf.constant(1.0 / 1.5, dtype=tf.float32)),
            (mean_kl_above_bound, lambda: tf.constant(1.5, dtype=tf.float32)),
        ],
        default=lambda: tf.constant(1.0, dtype=tf.float32),
        exclusive=True)

    new_adaptive_kl_beta = tf.maximum(
        self._adaptive_kl_beta * adaptive_kl_update_factor, 10e-16)
    tf.compat.v1.assign(self._adaptive_kl_beta, new_adaptive_kl_beta)

    if self._debug_summaries:
      tf.compat.v2.summary.scalar(
          name='adaptive_kl_update_factor',
          data=adaptive_kl_update_factor,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='mean_kl_divergence', data=mean_kl, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='adaptive_kl_beta',
          data=self._adaptive_kl_beta,
          step=self.train_step_counter)

    return self._adaptive_kl_beta
  
  def _handcraft_rewards(self, inference_input, zs):
    with tf.name_scope('handcraft_rewards'):
      def up(input):
        return -input[:, 1] - tf.abs(input[:, 0])
      def down(input):
        return input[:, 1] - tf.abs(input[:, 0])
      def right(input):
        return -input[:, 0] - tf.abs(input[:, 1])
      def left(input):
        return input[:, 0] - tf.abs(input[:, 1])
      index_0 = tf.squeeze(tf.where(zs[:, 0]==0), axis=-1)
      index_1 = tf.squeeze(tf.where(zs[:, 0]==1), axis=-1)
      index_2 = tf.squeeze(tf.where(zs[:, 0]==2), axis=-1)
      index_3 = tf.squeeze(tf.where(zs[:, 0]==3), axis=-1)
      slices_0 = tf.gather(inference_input, index_0)
      slices_1 = tf.gather(inference_input, index_1)
      slices_2 = tf.gather(inference_input, index_2)
      slices_3 = tf.gather(inference_input, index_3)
      rewards_list = []
      rewards_up = tf.map_fn(lambda x: up(x), slices_0, name="map0")
      rewards_list.append(rewards_up)
      rewards_down = tf.map_fn(lambda x: down(x), slices_1, name="map1")
      rewards_list.append(rewards_down)
      rewards_right = tf.map_fn(lambda x: right(x), slices_2, name="map2")
      rewards_list.append(rewards_right)
      rewards_left = tf.map_fn(lambda x: left(x), slices_3, name="map3")
      rewards_list.append(rewards_left)
      rewards = tf.concat(rewards_list, axis=0)
      common.generate_tensor_summaries(
            'index_0',
            index_0,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'index_1',
            index_1,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'index_2',
            index_2,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'index_3',
            index_3,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'rewards_up',
            rewards_up,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'rewards_down',
            rewards_down,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'rewards_right',
            rewards_right,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'rewards_left',
            rewards_left,
            step=self.train_step_counter)
      common.generate_tensor_summaries(
            'inference_input',
            inference_input,
            step=self.train_step_counter)
      return rewards
      
  def _cosine_similarity(self, inference_input, zs):
    horizon_length = inference_input.shape[1]
    if horizon_length is not None:
      batch_size = inference_input.shape[0]
      axis_right = tf.convert_to_tensor([[1, 0]]*batch_size, dtype=tf.float32)
      axis_right = tf.stack([axis_right] * horizon_length, axis=1)
      axis_up = tf.convert_to_tensor([[0, 1]]*batch_size, dtype=tf.float32)
      axis_up = tf.stack([axis_up] * horizon_length, axis=1)
      axis_left = tf.convert_to_tensor([[-1, 0]]*batch_size, dtype=tf.float32)
      axis_left = tf.stack([axis_left] * horizon_length, axis=1)
      axis_down = tf.convert_to_tensor([[0, -1]]*batch_size, dtype=tf.float32)
      axis_down = tf.stack([axis_down] * horizon_length, axis=1)
      axes = tf.zeros(inference_input.shape, dtype=tf.float32)
      zs_mask = tf.stack((zs, zs), axis=-1)
      axes = tf.where(zs_mask == 0, axis_right, axes)
      axes = tf.where(zs_mask == 1, axis_up, axes)
      axes = tf.where(zs_mask == 2, axis_left, axes)
      axes = tf.where(zs_mask == 3, axis_down, axes)
      norm_input = tf.math.l2_normalize(inference_input, axis=2)

      if norm_input.dtype == tf.float64:
          norm_input = tf.cast(norm_input, tf.float32)
      distances = tf.reduce_sum(tf.multiply(axes, norm_input), axis=2)
      likelihoods = (distances + 1.) / 2.
    else:
      likelihoods = 0.0
    return likelihoods

  def infer(self, inference_input, zs, step_type, weights):
    """Executes the inference."""
    if self._mock_inference:
        z_posteriors = self._cosine_similarity(inference_input, zs)
        log_z_posteriors = tf.math.log(z_posteriors)
    else:
      z_posteriors, _ = self._inference_net(
      inference_input, step_type, ())
      log_z_posteriors = common.log_probability(
          z_posteriors, zs, self._z_spec)

    if weights is not None:
        log_z_posteriors *= weights

    inference_loss = - tf.compat.v1.losses.compute_weighted_loss(
        log_z_posteriors)
    return inference_loss, log_z_posteriors, z_posteriors

  def inference_loss(self, time_steps, actions, next_time_steps, weights):
    """Computes the variational inference loss."""
    with tf.name_scope('inference_loss'):
      if self._mock_reward:
        return 0.0
      zs = time_steps.observation['z']
      next_observations = next_time_steps.observation['observation']
      rewards = next_time_steps.reward

      inference_input = time_steps.observation['observation'] - \
          next_time_steps.observation['observation']
      step_type = next_time_steps.step_type
      if self._infer_from_com:
        inference_input = inference_input[:, :, :2]

      inference_loss, log_z_posteriors, z_posteriors = self.infer(inference_input, zs, step_type, weights)

      if self._kl_posteriors_penalty is not None:
        if self._old_posteriors is None:
          kl_posteriors_loss = 0.0
        else:
          kl_posteriors_loss = tf.reduce_sum(z_posteriors.cross_entropy(self._old_posteriors) - z_posteriors.entropy())

        self._old_posteriors = z_posteriors
        latent_reconstruction_loss = tf.identity(inference_loss)
        inference_loss += self._kl_posteriors_penalty * kl_posteriors_loss

        tf.compat.v2.summary.scalar(
            'latent_reconstruction_loss',
            latent_reconstruction_loss,
            step=self.train_step_counter)
        tf.compat.v2.summary.scalar(
            'kl_posteriors_loss',
            kl_posteriors_loss,
            step=self.train_step_counter)
        
        
      inference_loss = self._inference_loss_coef * inference_loss

      if self._debug_summaries:
          
        tf.compat.v2.summary.scalar(
            'inference_loss',
            inference_loss,
            step=self.train_step_counter)
        common.generate_tensor_summaries(
            'log_z_posteriors',
            log_z_posteriors,
            step=self.train_step_counter)
        common.generate_tensor_summaries(
          'inference_input',
          inference_loss,
          step=self.train_step_counter)

        if isinstance(z_posteriors, tfp.distributions.Normal):
            common.generate_tensor_summaries(
                'z_posteriors_mean', z_posteriors.loc,
                self.train_step_counter)
            common.generate_tensor_summaries(
                'z_posteriors_stddev', z_posteriors.scale,
                self.train_step_counter)
        elif isinstance(z_posteriors, tfp.distributions.Categorical):
            common.generate_tensor_summaries(
                'z_posteriors_mode',
                z_posteriors.mode(),
                self.train_step_counter)
        else:
            common.generate_tensor_summaries(
                'z_posteriors',
                z_posteriors,
                self.train_step_counter)
            common.generate_tensor_summaries(
                'inference_dim_1',
                inference_input[0],
                self.train_step_counter)
            common.generate_tensor_summaries(
                'inference_dim_2',
                inference_input[1],
                self.train_step_counter)
            if not self._mock_inference:
              common.generate_tensor_summaries(
                  'z_posteriors_entropy', z_posteriors.entropy(),
                  self.train_step_counter)
    return inference_loss