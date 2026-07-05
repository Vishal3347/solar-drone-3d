import pygame
from stable_baselines3 import PPO
from env import SolarInspectionEnv

pygame.init()

WIDTH = 700
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solar Farm RL")

font = pygame.font.SysFont(None, 30)

env = SolarInspectionEnv()

model = PPO.load("solar_agent")

obs, _ = env.reset()

clock = pygame.time.Clock()

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    action, _ = model.predict(obs)

    obs, reward, done, truncated, info = env.step(action)

    screen.fill((255,255,255))

    for i, panel in enumerate(env.panel_locations):

        color = (255,0,0)

        if i in env.visited:
            color = (0,255,0)

        pygame.draw.circle(
            screen,
            color,
            panel,
            5
        )

    pygame.draw.circle(
        screen,
        (0,0,255),
        (int(obs[0]), int(obs[1])),
        8
    )

    battery_text = font.render(
        f"Battery: {info['battery']:.1f}",
        True,
        (0,0,0)
    )

    screen.blit(
        battery_text,
        (10,10)
    )

    panel_text = font.render(
        f"Visited: {info['visited']}/{info['total_panels']}",
        True,
        (0,0,0)
    )

    screen.blit(
        panel_text,
        (10,40)
    )

    pygame.display.flip()

    clock.tick(20)

    if done:
        print("Mission Complete")
        pygame.time.wait(3000)
        running = False

pygame.quit()