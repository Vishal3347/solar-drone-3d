import gymnasium as gym
from gymnasium import spaces
import numpy as np

from read_yolo import panel_locations


class SolarInspectionEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.size = 512

        self.panel_locations = panel_locations

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=self.size,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.drone = np.array([0, 0], dtype=np.float32)

        self.visited = set()

        self.battery = 100

        self.total_distance = 0

        return self.drone.copy(), {}

    def step(self, action):

        old_position = self.drone.copy()

        # Actions
        if action == 0:      # Up
            self.drone[1] -= 10

        elif action == 1:    # Down
            self.drone[1] += 10

        elif action == 2:    # Left
            self.drone[0] -= 10

        elif action == 3:    # Right
            self.drone[0] += 10

        # Stay inside map
        self.drone[0] = np.clip(
            self.drone[0],
            0,
            self.size
        )

        self.drone[1] = np.clip(
            self.drone[1],
            0,
            self.size
        )

        # Distance travelled
        distance = np.linalg.norm(
            self.drone - old_position
        )

        self.total_distance += distance

        # Battery usage
        self.battery -= 0.1

        reward = -1

        # Check panels
        for i, panel in enumerate(
            self.panel_locations
        ):

            panel = np.array(panel)

            d = np.linalg.norm(
                self.drone - panel
            )

            if d < 20 and i not in self.visited:

                self.visited.add(i)

                reward += 50

        done = False

        # All panels visited
        if len(self.visited) == len(
            self.panel_locations
        ):
            reward += 500
            done = True

        # Battery dead
        if self.battery <= 0:
            reward -= 100
            done = True

        info = {
            "battery": self.battery,
            "distance": self.total_distance,
            "visited": len(self.visited),
            "total_panels": len(
                self.panel_locations
            )
        }

        return (
            self.drone.copy(),
            reward,
            done,
            False,
            info
        )