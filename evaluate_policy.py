r"""Train and Eval DIAYN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
import time
import imageio
import IPython
import base64
import pdb


from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf
from tensorflow.python.framework import tensor_spec as ts
from collections import OrderedDict



from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import actor_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import greedy_policy

from agents.diayn import diayn_agent
from networks import critic_network
from utils import diayn_gym_env

from agents.lacma import lacma_agent
from policies import latent_actor_policy
from utils import hrl_metric_utils
from policies import skill_policy

flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_boolean(
    'hrl', False, 'If evaluating a hierarchical policy'
)
flags.DEFINE_integer(
    'steps_per_option', 50, 'Number of steps to take per high level skill'
)

FLAGS = flags.FLAGS

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

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

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

def evaluate_policy(root_dir):
    env_load_fn=suite_mujoco.load
    dim_z = 5
    categorical = True
    hrl = FLAGS.hrl
    steps_per_option = FLAGS.steps_per_option
    policy_fc_layers=(256, 256)
    env_steps = tf_metrics.EnvironmentSteps(prefix='Eval')
    _preprocessing_combiner = DictConcatenateLayer()
    global_step = tf.compat.v1.train.get_or_create_global_step()

    eval_env_name = "AntGoal-v1"
    tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(eval_env_name))
    eval_py_env = env_load_fn(eval_env_name)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    if categorical:
        z_spec = ts.BoundedTensorSpec(
        shape=[],
        dtype=tf.int64,
        minimum=0,
        maximum=dim_z-1,
        name='observation_z')
    else:
        z_spec = ts.BoundedTensorSpec(
            shape=[dim_z],
            dtype=tf.float32,
            minimum=-1.,
            maximum=1.,
            name='observation_z')
    
    if categorical:
        _preprocessing_combiner = OneHotConcatenateLayer(dim_z)
    else:
        _preprocessing_combiner = DictConcatenateLayer()

    eval_gen_net = actor_distribution_network.ActorDistributionNetwork(
            OrderedDict({"observation": observation_spec, "z": z_spec}),
            action_spec,
            fc_layer_params=policy_fc_layers,
            continuous_projection_net=normal_projection_net,
            preprocessing_combiner=_preprocessing_combiner,
            name='EvalNetwork')
    eval_gen_net_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_actor'),
            actor_net=eval_gen_net,
            global_step=global_step)
    eval_gen_net_checkpointer.initialize_or_restore()
    eval_latent_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            z_spec,
            fc_layer_params=policy_fc_layers,
            continuous_projection_net=normal_projection_net,
            name='LatentNetwork')
    eval_latent_net_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'lacma_latent_actor'),
            latent_actor_net=eval_latent_net,
            global_step=global_step)
    eval_latent_net_checkpointer.initialize_or_restore()
    
    if hrl:
        high_level_eval_policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec,
            action_spec=z_spec,
            actor_network=eval_latent_net,
            training=False)
        low_level_eval_policy = skill_policy.SkillPolicy(
            time_step_spec=time_step_spec,
            z_spec=z_spec,
            action_spec=action_spec,
            generator_network=eval_gen_net,
            steps_per_option=steps_per_option,
            training=False)
        high_level_policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'hrl_high_level_policy'),
            policy=high_level_eval_policy,
            global_step=global_step)
        low_level_policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'hrl_low_level_policy'),
            policy=low_level_eval_policy,
            global_step=global_step)
        high_level_policy_checkpointer.initialize_or_restore()
        low_level_policy_checkpointer.initialize_or_restore()
    else:
        eval_policy_distribution = latent_actor_policy.LatentActorPolicy(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                actor_network=eval_latent_net,
                generator_network=eval_gen_net,
                training=False)
        eval_policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'lacma_policy'),
                policy=eval_policy_distribution,
                global_step=global_step)
        eval_policy_checkpointer.initialize_or_restore()
        eval_policy = greedy_policy.GreedyPolicy(eval_policy_distribution)

    print("Loaded evaluation policy from checkpoint")
    num_eval_episodes = 30
    summary_interval=1000
    summaries_flush_secs=10
    debug_summaries=False
    summarize_grads_and_vars=False
    eval_metrics_callback=None
    summary_writer = tf.compat.v2.summary.create_file_writer(
        root_dir, flush_millis=summaries_flush_secs * 1000)
    summary_writer.set_as_default()
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]
    if hrl:
        results = hrl_metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                high_level_eval_policy,
                low_level_eval_policy,
                num_episodes=num_eval_episodes,
                train_step=env_steps.result(),
                summary_writer=summary_writer,
                summary_prefix='Eval',
        )
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, env_steps.result())
        hrl_metric_utils.log_metrics(eval_metrics)
    else:
        results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=env_steps.result(),
                summary_writer=summary_writer,
                summary_prefix='Eval',
            )
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, env_steps.result())
        metric_utils.log_metrics(eval_metrics)

    print("Creating video")
    rollout_path = root_dir + "/rollouts"
    if not path.exists(rollout_path):
        os.mkdir(rollout_path)

    video_filename = root_dir + '/rollouts/lacma_' + eval_env_name[:-3] + '.mp4'
    path_len = 1000
    num_eps = 1
    with imageio.get_writer(video_filename, fps=60) as video:
        for _ in range(num_eps):
            _time_step = eval_py_env.reset()
            video.append_data(eval_py_env.render())
            steps = 0
            _z = None
            while steps < path_len:                    
                if hrl:
                    if steps % steps_per_option == 0:
                        _z = high_level_eval_policy.action(_time_step).action
                        if _z.numpy() == 0:
                            skill = "up"
                        elif _z.numpy() == 1:
                            skill = "down"
                        elif _z.numpy() == 2: 
                            skill = "right"
                        elif _z.numpy() == 3:
                            skill = "left"
                        elif _z.numpy() == 4:
                            skill = "stop"
                        print("Skill at step {}: {}".format(steps, skill))
                    action_step = low_level_eval_policy.action(_time_step, _z)
                else:
                    action_step = eval_policy.action(_time_step)
                _time_step = eval_py_env.step(action_step.action.numpy())
                video.append_data(eval_py_env.render())
                steps += 1
        embed_mp4(video_filename)

    
def main(_):
    tf.compat.v1.enable_v2_behavior()
    evaluate_policy(FLAGS.root_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
