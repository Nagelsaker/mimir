import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class TestMJ(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.viewer = None
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.zeros(3)
        mujoco_env.MujocoEnv.__init__(
            self, "/home/simon/catkin_ws/src/mimir/xml/open_manipulator_and_lever.xml", 2
        )


if __name__ == "__main__":
    test = TestMJ()