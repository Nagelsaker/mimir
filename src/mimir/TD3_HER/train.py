from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from open_manipulator_rl_environments.task_environments.lever_pull_task_mujoco import OpenManipulatorMujocoLeverPullEnvironment

import os
import gym
import arguments


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

def launch(env, model_class, model_path):
    # env = OpenManipulatorLeverPullEnvironment()

    args = arguments.get_args()

    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

    # If True the HER transitions will get sampled online
    online_sampling = True
    # Time limit for the episodes
    max_episode_length = args.n_cycles
    total_timesteps = 1e6
    action_noise = NormalActionNoise(0, 0.2)

    # Initialize the model
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        train_freq=(2, "episode"),
        # gradient_steps = 50,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4, #4,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
            max_episode_length=max_episode_length,
        ),
        learning_rate=args.lr_actor,
        buffer_size=int(1e6),
        # batch_size=50,
        tau=args.polyak,
        gamma=args.gamma,
        action_noise=action_noise,
        verbose=1,
    )

    path_to_log = f"{os.path.dirname(os.path.abspath(__file__))}/evals/"
    model.learn(total_timesteps, n_eval_episodes=30, log_interval=50, eval_log_path=path_to_log)

    model.save(model_path)
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env


def testModel(env, model_class, model_path):
    model = model_class.load(model_path, env=env)

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        if done:
            obs = env.reset()                   


if __name__ == "__main__":
    env = gym.make("OpenManipulator_lever_pull_task_mujoco-v0")
    env_params = get_env_params(env)

    model_class = TD3  #     works also with SAC, DDPG and TD3
    model_path = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models/model1"
    launch(env, model_class, model_path)
    # testModel(env, model_class, model_path)