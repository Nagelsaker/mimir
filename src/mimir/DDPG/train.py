import numpy as np
import gym
import rospy
import mujoco_py
import os, sys, time
from gym.envs.mujoco.mujoco_env import MujocoEnv
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from ddpg_agent import ddpg_agent

# Important! Does not work without including the environment
from open_manipulator_rl_environments.task_environments.lever_pull_task_mujoco import OpenManipulatorMujocoLeverPullEnvironment

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # env = OpenManipulatorLeverPullEnvironment()
    env = gym.make("OpenManipulator_lever_pull_task_mujoco-v0")
    # env = gym.make("Pendulum-v1")
    rospy.loginfo("LOADED!")
    env_params = get_env_params(env)

    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

def visualizeTraining(idx):
    import matplotlib.pyplot as plt
    path_to_metadata = f"{os.path.dirname(os.path.abspath(__file__))}/meta_data/success_reward.npy"
    meta = np.load(path_to_metadata)
    plt.plot(meta[:,idx])
    plt.show()

if __name__ == '__main__':
    # rospy.init_node('train_net_node')
    # args = get_args()
    # launch(args)
    visualizeTraining(int(sys.argv[1]))
    

