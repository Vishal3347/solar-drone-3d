
"""
Tests a trained agent (solar_agent.zip, produced by train.py) across
several episodes. Each episode = one drone mission inspecting all
panels detected in one real drone image from the dataset.
"""
 
from stable_baselines3 import PPO
from env import SolarInspectionEnv
 
NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 1000
 
env = SolarInspectionEnv()
model = PPO.load("solar_agent")
 
completed = 0
completion_rates = []
 
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
 
    for step in range(MAX_STEPS_PER_EPISODE):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
 
        if done:
            break
 
    total = info["total_panels"]
    visited = info["visited"]
    rate = (visited / total) if total else 1.0
    completion_rates.append(rate)
    if rate == 1.0:
        completed += 1
 
    print(
        f"Episode {ep}: image={info['image']} | "
        f"panels visited={visited}/{total} ({rate*100:.0f}%) | "
        f"steps={step + 1} | battery left={info['battery']:.1f}"
    )
 
print()
print(f"Fully completed missions: {completed}/{NUM_EPISODES}")
print(f"Average completion rate: {sum(completion_rates)/len(completion_rates)*100:.1f}%")
 
