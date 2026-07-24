import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from read_yolo import panel_locations_by_image
import config


class SolarInspectionEnv(gym.Env):
    """
    Each episode = one drone inspecting ONE real photo's worth of solar
    panels (2-15 panels per image, matching the actual dataset). This
    matches reality: a drone image only shows one physical section of
    the farm, so panels from 379 different photos must NOT be treated
    as one shared grid -- each episode picks one image and the drone's
    job is to visit every panel detected in that image.
    """

    def __init__(self):
        super().__init__()

        self.size = config.ENV_SIZE
        self.images = list(panel_locations_by_image.keys())

        self.action_space = spaces.Discrete(4)

        # Observation: [drone_x, drone_y, dx_to_nearest_unvisited_panel, dy_to_nearest_unvisited_panel]
        # The direction vector is what makes this solvable -- without it the
        # agent has no way to know where any panel is and can only memorize
        # a single fixed layout (which breaks the moment images vary per episode).
        self.observation_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(4,),
            dtype=np.float32
        )

    def _get_obs(self):
        dx, dy = 0.0, 0.0
        best_dist = None

        for i, panel in enumerate(self.panel_locations):
            if i in self.visited:
                continue
            panel = np.array(panel, dtype=np.float32)
            d = np.linalg.norm(self.drone - panel)
            if best_dist is None or d < best_dist:
                best_dist = d
                dx, dy = panel[0] - self.drone[0], panel[1] - self.drone[1]

        return np.array([self.drone[0], self.drone[1], dx, dy], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.images:
            image_name = random.choice(self.images)
            self.panel_locations = panel_locations_by_image[image_name]
            self.current_image = image_name
        else:
            self.panel_locations = []
            self.current_image = None

        self.drone = np.array([0, 0], dtype=np.float32)

        self.visited = set()

        self.battery = 100

        self.total_distance = 0

        return self._get_obs(), {}

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
        if len(self.panel_locations) > 0 and len(self.visited) == len(
            self.panel_locations
        ):
            reward += 500
            done = True

        # Battery dead
        if self.battery <= 0:
            reward -= 100
            done = True

        info = {
            "image": self.current_image,
            "battery": self.battery,
            "distance": self.total_distance,
            "visited": len(self.visited),
            "total_panels": len(
                self.panel_locations
            )
        }

        return (
            self._get_obs(),
            reward,
            done,
            False,
            info
        )