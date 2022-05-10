import numpy as np
import os, sys
import gym
import rospy
from arguments import get_args
from ddpg_agent import ddpg_agent
from train import get_env_params

# Important! Does not work without including the environment
from open_manipulator_rl_environments.task_environments.lever_pull_task_mujoco import OpenManipulatorMujocoLeverPullEnvironment



def evalTraining(name, n_evals=5):
    import matplotlib.pyplot as plt

    env = gym.make("OpenManipulator_lever_pull_task_mujoco-v0")
    rospy.loginfo("LOADED!")
    env_params = get_env_params(env)
    # args = get_args()
    args = rospy.get_param("/mimir/DDPG/")

    load_model_path = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models/OpenManipulator_lever_pull_task-v0/model{name}.pt"
    ddpg_model = ddpg_agent(args, env, env_params, load_model_path)

    avg_success_rate = 0
    avg_avg_reward = 0
    for i in range(n_evals):
        success_rate, avg_reward = ddpg_model._eval_agent()
        print(f"success rate: {success_rate}, avg reward: {avg_reward}")
        avg_success_rate += success_rate
        avg_avg_reward += avg_reward
    avg_success_rate /= n_evals
    avg_avg_reward /= n_evals
    print(f"avg success rate: {avg_success_rate}, avg avg reward: {avg_avg_reward}")

    
if __name__ == '__main__':
    #print(f"{int(sys.argv[1])}")
    # evalTraining(int(sys.argv[1]), sys.argv[2])
    evalTraining("")
    # visualizeTraining(0, "")
    

