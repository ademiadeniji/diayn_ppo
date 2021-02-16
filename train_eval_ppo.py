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

r"""Train and Eval PPO.

To run:

```bash
tensorboard --logdir $HOME/tmp/ppo/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/ppo/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/ppo/gym/HalfCheetah-v2/ \
  --logtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pdb

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from agents.diayn import ppo_diayn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tensorflow.python.framework import tensor_spec as tspec

from utils import diayn_gym_env


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'Plane-v1', 'Name of an environment')
flags.DEFINE_integer('replay_buffer_capacity', 1001,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_environments', 30,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_environment_steps', 50000000,
                     'Number of environment steps to run before finishing.')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'collect_episodes_per_iteration', 30,
    'The number of episodes to take in the environment before '
    'each update. This is the total across all parallel '
    'environments.')
flags.DEFINE_integer('num_eval_episodes', 30,
                     'The number of episodes to run eval on.')
flags.DEFINE_boolean('use_rnns', False,
                     'If true, use RNN for policy and value function.')
flags.DEFINE_float('entropy_regularization', 0.0, 'Amount to entropy regularize policy')
flags.DEFINE_boolean(
    'mock_inference', False, 'If using mocked inference network'
)
flags.DEFINE_boolean(
    'mock_reward', False, 'If using handcrafted reward'
)
flags.DEFINE_boolean(
    'l2_distance', False, 'If using l2 distance from origin loss'
)
flags.DEFINE_integer(
    'rl_steps', None, 'Number of steps to train actor'
)
flags.DEFINE_integer(
    'inference_steps', None, 'Number of steps to train inference'
)
flags.DEFINE_float(
    'kl_posteriors_penalty', None, 'How much to penalize shift in posteriors'
)
FLAGS = flags.FLAGS

class DictConcatenateLayer(tf.keras.layers.Layer):

    def __init__(self, axis=-1):
        super(DictConcatenateLayer, self).__init__()
        self._concat_layer = tf.keras.layers.Concatenate(axis=axis)
        self._axis = -1

    def call(self, inputs):
        list_inputs = list(inputs.values())
        return self._concat_layer(list_inputs)

    def get_config(self):
        return {'axis': self._axis}


class OneHotConcatenateLayer(DictConcatenateLayer):

    def __init__(self, depth, axis=-1):
        super(OneHotConcatenateLayer, self).__init__(axis=axis)
        self._depth = depth

    def call(self, inputs):
        one_hot = tf.one_hot(inputs['z'], depth=self._depth)
        list_inputs = [inputs['observation'], one_hot]
        return self._concat_layer(list_inputs)

    def get_config(self):
        return {'depth': self._depth, 'axis': self._axis}

@gin.configurable
def std_clip_transform(stddevs):
    stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 2),
                                    stddevs)
    return tf.exp(stddevs)

@gin.configurable
def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.001):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        init_means_output_factor=init_means_output_factor,
        std_transform=std_clip_transform,
        scale_distribution=True,
        state_dependent_std=True)

@gin.configurable
def train_eval(
    root_dir,
    env_name=None,
    env_load_fn=suite_mujoco.load,
    random_seed=0,
    # TODO(b/127576522): rename to policy_fc_layers.
    actor_fc_layers=(200, 100),
    value_fc_layers=(200, 100),
    inference_fc_layers=(200, 100),
    use_rnns=None,
    dim_z=4,
    categorical=True,
    # Params for collect
    num_environment_steps=10000000,
    collect_episodes_per_iteration=30,
    num_parallel_environments=30,
    replay_buffer_capacity=1001,  # Per-environment
    # Params for train
    num_epochs=25,
    learning_rate=1e-4,
    entropy_regularization=None,
    kl_posteriors_penalty=None,
    mock_inference=None,
    mock_reward=None,
    l2_distance=None,
    rl_steps=None,
    inference_steps=None,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=1000,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=10000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=1,
    use_tf_functions=True,
    debug_summaries=False,
    summarize_grads_and_vars=False):
  """A simple train and eval for PPO."""
  if root_dir is None:
    raise AttributeError('train_eval requires a root_dir.')

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf.compat.v1.set_random_seed(random_seed)

    def _env_load_fn(env_name):
        diayn_wrapper = (
            lambda x: diayn_gym_env.DiaynGymEnv(x, dim_z, categorical)
        )
        return env_load_fn(
            env_name,
            gym_env_wrappers=[diayn_wrapper],
        )

    eval_tf_env = tf_py_environment.TFPyEnvironment(_env_load_fn(env_name))
    if num_parallel_environments == 1:
        py_env = _env_load_fn(env_name)
    else:
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: _env_load_fn(env_name)] * num_parallel_environments)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    augmented_time_step_spec = tf_env.time_step_spec()
    augmented_observation_spec = augmented_time_step_spec.observation
    observation_spec = augmented_observation_spec['observation']
    z_spec = augmented_observation_spec['z']
    reward_spec = augmented_time_step_spec.reward
    action_spec = tf_env.action_spec()
    time_step_spec = ts.time_step_spec(observation_spec)
    infer_from_com = False
    if env_name == "AntRandGoalEval-v1":
        infer_from_com = True
    if infer_from_com:
        input_inference_spec = tspec.BoundedTensorSpec(
            shape=[2],
            dtype=tf.float64,
            minimum=-1.79769313e+308,
            maximum=1.79769313e+308,
            name='body_com')
    else: 
        input_inference_spec = observation_spec

    if tensor_spec.is_discrete(z_spec):
        _preprocessing_combiner = OneHotConcatenateLayer(dim_z)
    else:
        _preprocessing_combiner = DictConcatenateLayer()

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    
    if use_rnns:
      actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
          augmented_observation_spec,
          action_spec,
          preprocessing_combiner=_preprocessing_combiner,
          input_fc_layer_params=actor_fc_layers,
          output_fc_layer_params=None)
      value_net = value_rnn_network.ValueRnnNetwork(
          augmented_observation_spec,
          preprocessing_combiner=_preprocessing_combiner,
          input_fc_layer_params=value_fc_layers,
          output_fc_layer_params=None)
    else:
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          augmented_observation_spec,
          action_spec,
          preprocessing_combiner=_preprocessing_combiner,
          fc_layer_params=actor_fc_layers,
          name="actor_net")
      value_net = value_network.ValueNetwork(
          augmented_observation_spec, 
          preprocessing_combiner=_preprocessing_combiner,
          fc_layer_params=value_fc_layers,
          name="critic_net")
    inference_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=input_inference_spec,
        output_tensor_spec=z_spec,
        fc_layer_params=inference_fc_layers,
        continuous_projection_net=normal_projection_net,
        name="inference_net")
    
    tf_agent = ppo_diayn_agent.PPODiaynAgent(
        augmented_time_step_spec,
        action_spec,
        z_spec,
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        inference_net=inference_net,
        num_epochs=num_epochs,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
        entropy_regularization=entropy_regularization,
        kl_posteriors_penalty=kl_posteriors_penalty,
        mock_inference=mock_inference,
        mock_reward=mock_reward,
        infer_from_com=infer_from_com,
        l2_distance=l2_distance,
        rl_steps=rl_steps,
        inference_steps=inference_steps)
    tf_agent.initialize()   
    
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(
            batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=num_parallel_environments),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity)

    actor_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'diayn_actor'),
        actor_net=actor_net,
        global_step=global_step)
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'diayn_policy'),
        policy=eval_policy,
        global_step=global_step)
    saved_model = policy_saver.PolicySaver(
        eval_policy, train_step=global_step)
    rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)
    inference_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_inference'),
            inference_net=inference_net,
            global_step=global_step)
        

    actor_checkpointer.initialize_or_restore()
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()
    inference_checkpointer.initialize_or_restore()

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration)

    # option_length = 200
    # if env_name == "Plane-v1":
    #     option_length = 10
    # dataset = replay_buffer.as_dataset(
    #         num_parallel_calls=3, sample_batch_size=num_parallel_environments,
    #         num_steps=option_length)
    # iterator_dataset = iter(dataset)

    def train_step():
      trajectories = replay_buffer.gather_all()
    #   trajectories, _ = next(iterator_dataset)
      return tf_agent.train(experience=trajectories)

    if use_tf_functions:
      # TODO(b/123828980): Enable once the cause for slowdown was identified.
      collect_driver.run = common.function(collect_driver.run, autograph=False)
      tf_agent.train = common.function(tf_agent.train, autograph=False)
      train_step = common.function(train_step)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_steps_metric.result() < num_environment_steps:
      global_step_val = global_step.numpy()
      if global_step_val % eval_interval == 0:
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )

      start_time = time.time()
      collect_driver.run()
      collect_time += time.time() - start_time

      start_time = time.time()
      total_loss, _ = train_step()
      replay_buffer.clear()
      train_time += time.time() - start_time

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=step_metrics)

      if global_step_val % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step_val, total_loss)
        steps_per_sec = (
            (global_step_val - timed_at_step) / (collect_time + train_time))
        logging.info('%.3f steps/sec', steps_per_sec)
        logging.info('collect_time = {}, train_time = {}'.format(
            collect_time, train_time))
        with tf.compat.v2.summary.record_if(True):
          tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=steps_per_sec, step=global_step)

        if global_step_val % train_checkpoint_interval == 0:
          train_checkpointer.save(global_step=global_step_val)
          inference_checkpointer.save(global_step=global_step_val)
          actor_checkpointer.save(global_step=global_step_val)
          rb_checkpointer.save(global_step=global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
          policy_checkpointer.save(global_step=global_step_val)
          saved_model_path = os.path.join(
              saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
          saved_model.save(saved_model_path)

        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

    # One final eval before exiting.
    metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  train_eval(
      FLAGS.root_dir,
      env_name=FLAGS.env_name,
      use_rnns=FLAGS.use_rnns,
      num_environment_steps=FLAGS.num_environment_steps,
      collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
      num_parallel_environments=FLAGS.num_parallel_environments,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      num_epochs=FLAGS.num_epochs,
      num_eval_episodes=FLAGS.num_eval_episodes,
      entropy_regularization=FLAGS.entropy_regularization,
      kl_posteriors_penalty=FLAGS.kl_posteriors_penalty,
      mock_inference=FLAGS.mock_inference,
      mock_reward=FLAGS.mock_reward,
      l2_distance=FLAGS.l2_distance,
      rl_steps=FLAGS.rl_steps,
      inference_steps=FLAGS.inference_steps)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)