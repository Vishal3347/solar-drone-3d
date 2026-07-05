from stable_baselines3 import PPO
from env import SolarInspectionEnv

env = SolarInspectionEnv()

model = PPO.load("solar_agent")

obs, _ = env.reset()

for step in range(1000):
    action, _ = model.predict(obs)

    obs, reward, done, truncated, info = env.step(action)

    print(
        f"Step {step}: Position={obs}, Reward={reward}"
    )

    if done:
        print("All panels inspected!")
        break