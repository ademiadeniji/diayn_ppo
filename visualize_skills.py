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
import copy
import d4rl
import numpy as np

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf
from tensorflow.python.framework import tensor_spec as ts
from collections import OrderedDict
import matplotlib.pyplot as plt



from tf_agents.policies import epsilon_greedy_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import normal_projection_network
from tf_agents.policies import actor_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from agents.diayn import diayn_agent
from networks import critic_network
from utils import diayn_gym_env_fixed
from networks import actor_distribution_network
# from tf_agents.networks import actor_distribution_network

from agents.lacma import lacma_agent
from policies import latent_actor_policy

flags.DEFINE_integer(
    'dim_z', 5, 'Embedding size.'
)
flags.DEFINE_boolean(
    'mask_xy', True, 'Whether to mask xy position information from agent'
)
flags.DEFINE_float(
    'skill_epsilon', 0.5, 'Probability of sampling random skill'
)
flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'AntSpace-v1', 'Name of an environment')
# flags.DEFINE_string('env_name', 'antmaze-umaze-v0', 'Name of an environment')


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

def evaluate_policy():
    env_load_fn=suite_mujoco.load
    categorical = True
    FLAGS = flags.FLAGS
    dim_z = FLAGS.dim_z
    mask_xy = FLAGS.mask_xy
    eval_env_name = FLAGS.env_name
    skill_epsilon = FLAGS.skill_epsilon
    epsilon = 0.75
    epsilon_greedy = False
    state_noise = False
    action_noise = False
    skill_randomization = True
    plot_actions = False
    

    def _env_load_fn(env_name):
        diayn_wrapper = (
            lambda x: diayn_gym_env_fixed.DiaynGymEnvFixed(x, dim_z, categorical)
        )
        return env_load_fn(
            env_name,
            gym_env_wrappers=[diayn_wrapper],
        )

    root_dir = FLAGS.root_dir
    policy_fc_layers=(256, 256)
    env_steps = tf_metrics.EnvironmentSteps(prefix='Eval')
    _preprocessing_combiner = DictConcatenateLayer()
    global_step = tf.compat.v1.train.get_or_create_global_step()

    if eval_env_name == "Plane-v1":
        make_video = False
    else:
        make_video = True
    tf_env = tf_py_environment.TFPyEnvironment(_env_load_fn(eval_env_name))
    eval_py_env = _env_load_fn(eval_env_name)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()
    augmented_time_step_spec = tf_env.time_step_spec()
    augmented_observation_spec = augmented_time_step_spec.observation
    z_spec = augmented_observation_spec["z"]

    if tensor_spec.is_discrete(z_spec):
        _preprocessing_combiner = OneHotConcatenateLayer(dim_z)
    else:
        _preprocessing_combiner = DictConcatenateLayer()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
            augmented_observation_spec,
            action_spec,
            fc_layer_params=policy_fc_layers,
            continuous_projection_net=normal_projection_net,
            preprocessing_combiner=_preprocessing_combiner,
            mask_xy=mask_xy,
            name='EvalNetwork')

    generator_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_actor'),
            actor_net=actor_net,
            global_step=global_step
        )
    if generator_checkpointer.checkpoint_exists:
        generator_checkpointer.initialize_or_restore()
    else:
        generator_checkpointer.initialize_or_restore()
        print("No low-level actor checkpoint exists...training from scratch")
    # Re-purpose the restored actor network
    generator_net = actor_net

    eval_policy = actor_policy.ActorPolicy(
            time_step_spec=augmented_time_step_spec,
            action_spec=action_spec,
            actor_network=generator_net,
            training=False)
    
    print("Loaded evaluation policy")

    if make_video:
        print("Creating video")
        skill_path = root_dir + "/skills"
        action_path = root_dir + "/actions"
        if not path.exists(skill_path):
            os.mkdir(skill_path)
        if not path.exists(action_path):
            os.mkdir(action_path)
        
        color_wheel = ['b', 'r', 'g', 'c', 'm']
        for i in range(5):
            for runs in range(dim_z):
                xs_list = []
                ys_list = []
                actions_list = {new_list: [] for new_list in range(8)} 
                video_filename = root_dir + '/skills/' + str(i+1) + eval_env_name[:-3] + '.mp4'
                skill_plot_filename = root_dir + '/skills/' + str(i+1) + eval_env_name[:-3] + 'eps' \
                     + str(skill_epsilon) + '.png'
                action_plot_filename = root_dir + '/actions/' + str(i+1) + eval_env_name[:-3] + '.png'
                path_len = 200
                num_eps = 1
                print_interval = 20
                action_interval = 10
                skill_sample_interval = 20
                print("skill {}".format(i))
                with imageio.get_writer(video_filename, fps=60) as video:
                    for _ in range(num_eps):
                        if categorical:
                            eval_py_env.set_z(i)
                        else:
                            skill = [0] * dim_z
                            skill[i] = 1
                            eval_py_env.set_z(skill)
                        _time_step = eval_py_env.reset()
                        print("{} {}".format(_time_step.observation["observation"][:2][0], _time_step.observation["observation"][:2][1]))
                        video.append_data(eval_py_env.render())
                        steps = 0
                        while steps < path_len:
                            if skill_randomization:
                                if steps % skill_sample_interval == 0:
                                    if np.random.random() < skill_epsilon:
                                        sampled_skill = np.random.choice(dim_z)
                                        eval_py_env.set_z(sampled_skill)
                                        print("randomly sampled skill: {}".format(sampled_skill))
                                    else:
                                        eval_py_env.set_z(i)
                                        print("stuck with skill {}".format(i))
                            if state_noise:
                                _time_step.observation["observation"] = np.random.normal(_time_step.observation["observation"], scale=0.25)
                            if epsilon_greedy:
                                sample = np.random.random_sample()            
                                if sample < epsilon:
                                    action = tensor_spec.sample_spec_nest(action_spec).numpy()
                                else:
                                    action = eval_policy.action(_time_step).action.numpy()
                            else: 
                                action = eval_policy.action(_time_step).action.numpy()
                            if steps % action_interval == 0:
                                for index in range(action.shape[0]):
                                    actions_list[index].append(action[index])
                            if action_noise:
                                noisy_action = np.random.normal(action, scale=1.0)
                                _time_step = eval_py_env.step(noisy_action)
                            else:
                                _time_step = eval_py_env.step(action)
                            if steps % print_interval == 0: 
                                print("{} {}".format(_time_step.observation["observation"][:2][0], _time_step.observation["observation"][:2][1]))
                                xs_list.append(_time_step.observation["observation"][:2][0])
                                ys_list.append(_time_step.observation["observation"][:2][1])
                            video.append_data(eval_py_env.render())
                            steps += 1
                    embed_mp4(video_filename)
                    if plot_actions:
                        plt.ylim(-1, 1)
                        for i in range(8):
                            plt.plot(range(int(200/action_interval)), actions_list[i])
                    plt.xlim(-25, 25)
                    plt.ylim(-25, 25)
                    plt.plot(xs_list, ys_list, color_wheel[i])
            plt.savefig(skill_plot_filename)
            if plot_actions: plt.savefig(action_plot_filename)
    else:
        print("Rendering plane skills")
        skill_path = root_dir + "/skills"
        if not path.exists(skill_path):
            os.mkdir(skill_path)

        for i in range(0, dim_z, 1):
            path_len = 10
            num_eps = 1
            xs_list = []
            ys_list = []
            skill_plot_filename = root_dir + '/skills/' + str(i+1) + eval_env_name[:-3] + '.png'
            for _ in range(num_eps):
                if categorical:
                    eval_py_env.set_z(i)
                    print("skill {}".format(i+1))
                else:
                    skill = [0] * dim_z
                    skill[i] = 1
                    eval_py_env.set_z(skill)
                _time_step = eval_py_env.reset()
                eval_py_env.render()
                steps = 0
                while steps < path_len:
                    xs_list.append(_time_step.observation["observation"][0])
                    ys_list.append(_time_step.observation["observation"][1])
                    action_step = eval_policy.action(_time_step)
                    _time_step = eval_py_env.step(action_step.action.numpy())
                    eval_py_env.render()
                    steps += 1
            plt.xlim(-100, 100)
            plt.ylim(-100, 100)
            plt.plot(xs_list, ys_list)
        plt.savefig(skill_plot_filename)

def main(_):
    tf.compat.v1.enable_v2_behavior()
    evaluate_policy()


if __name__ == '__main__':
    app.run(main)
