import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from multiprocessing import Pool

# Pygame setup
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
GRID_SIZE = 100
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Constants
NUM_AGENTS = 100
NUM_EPOCHS = 100
ENERGY_COST = 0.1
NUM_PROCESSES = 4  # Number of parallel processes

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 50
        self.money = 0

    def move(self, dx, dy):
        self.x = max(0, min(GRID_SIZE - 1, self.x + dx))
        self.y = max(0, min(GRID_SIZE - 1, self.y + dy))
        self.energy -= np.hypot(dx, dy) * ENERGY_COST

    def eat(self):
        if self.money >= 5:
            self.energy += 10
            self.money -= 5

    def work(self):
        if self.energy >= 5:
            self.money += 10
            self.energy -= 5

    def is_alive(self):
        return self.energy > 0

def spawn_jobs_and_food(grid):
    for _ in range(50):
        grid[random.randint(0, GRID_SIZE - 1)][random.randint(0, GRID_SIZE - 1)] = 'food'
        grid[random.randint(0, GRID_SIZE - 1)][random.randint(0, GRID_SIZE - 1)] = 'job'

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.fc(x)

def agent_update(agent, model, grid):
    if not agent.is_alive():
        return agent

    state = np.array([agent.x, agent.y, agent.energy, agent.money])
    action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

    if action == 0:
        agent.move(1, 0)
    elif action == 1:
        agent.move(-1, 0)
    elif action == 2:
        agent.move(0, 1)
    else:
        agent.move(0, -1)

    reward = 0
    if grid[agent.x][agent.y] == 'food':
        agent.eat()
        reward = 10
    elif grid[agent.x][agent.y] == 'job':
        agent.work()
        reward = 10
    else:
        reward = -1

    return agent

def train_dqn():
    model = DQN()
    optimizer = optim.Adam(model.parameters())
    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99

    for epoch in range(NUM_EPOCHS):
        agents = [Agent(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)) for _ in range(NUM_AGENTS)]
        grid = [['empty'] * GRID_SIZE for _ in range(GRID_SIZE)]
        spawn_jobs_and_food(grid)
        epoch_alive_agents = NUM_AGENTS

        with Pool(NUM_PROCESSES) as pool:
            while any(agent.is_alive() for agent in agents):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                agents = pool.starmap(agent_update, [(agent, model, grid) for agent in agents])

                # Visualization
                screen.fill((0, 0, 0))
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):
                        if grid[x][y] == 'food':
                            pygame.draw.rect(screen, (0, 255, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                        elif grid[x][y] == 'job':
                            pygame.draw.rect(screen, (0, 0, 255), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

                for agent in agents:
                    if agent.is_alive():
                        pygame.draw.rect(screen, (255, 0, 0), (agent.x * CELL_SIZE, agent.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    else:
                        epoch_alive_agents -= 1

                # Display stats
                alive_agents = sum(agent.is_alive() for agent in agents)
                avg_energy = np.mean([agent.energy for agent in agents if agent.is_alive()])
                avg_money = np.mean([agent.money for agent in agents if agent.is_alive()])

                stats_text = f'Epoch: {epoch + 1}/{NUM_EPOCHS} - Alive Agents: {alive_agents} - Avg Energy: {avg_energy:.2f} - Avg Money: {avg_money:.2f}'
                stats_surface = font.render(stats_text, True, (255, 255, 255))
                screen.blit(stats_surface, (10, 10))

                pygame.display.flip()

if __name__ == "__main__":
    train_dqn()
    pygame.quit()
