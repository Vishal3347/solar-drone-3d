from stable_baselines3 import PPO
from env import SolarInspectionEnv

env = SolarInspectionEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=100000
)

model.save("solar_agent")

print("Training Finished")