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
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 1200
GRID_SIZE = 100
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Constants
NUM_AGENTS = 100
NUM_EPOCHS = 100
ENERGY_COST = 0.1
NUM_PROCESSES = 24  # Number of parallel processes

class Unit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 50
        self.money = 0

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Ensure units stay within bounds
        if 0 <= new_x < GRID_SIZE:
            self.x = new_x
        if 0 <= new_y < GRID_SIZE:
            self.y = new_y
        
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

def unit_update(unit, model, grid):
    if not unit.is_alive():
        return unit

    state = np.array([unit.x, unit.y, unit.energy, unit.money])
    action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

    if action == 0:
        unit.move(1, 0)
    elif action == 1:
        unit.move(-1, 0)
    elif action == 2:
        unit.move(0, 1)
    else:
        unit.move(0, -1)

    reward = 0
    if grid[unit.x][unit.y] == 'food':
        unit.eat()
        reward = 10
    elif grid[unit.x][unit.y] == 'job':
        unit.work()
        reward = 10
    else:
        reward = -1

    return unit

def train_dqn():
    model = DQN()
    optimizer = optim.Adam(model.parameters())
    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if os.path.exists('model.pth'):
            model.load_state_dict(torch.load('model.pth'))

        for epoch in range(NUM_EPOCHS):
            units = [Unit(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)) for _ in range(NUM_AGENTS)]
            grid = [['empty'] * GRID_SIZE for _ in range(GRID_SIZE)]
            spawn_jobs_and_food(grid)
            epoch_alive_units = NUM_AGENTS

            with Pool(NUM_PROCESSES) as pool:
                while any(unit.is_alive() for unit in units):
                    units = pool.starmap(unit_update, [(unit, model, grid) for unit in units])

                    # Visualization
                    screen.fill((0, 0, 0))
                    for x in range(GRID_SIZE):
                        for y in range(GRID_SIZE):
                            if grid[x][y] == 'food':
                                pygame.draw.rect(screen, (0, 255, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                            elif grid[x][y] == 'job':
                                pygame.draw.rect(screen, (0, 0, 255), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

                    for unit in units:
                        if unit.is_alive():
                            pygame.draw.rect(screen, (255, 0, 0), (unit.x * CELL_SIZE, unit.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                        else:
                            epoch_alive_units -= 1

                    # Display stats
                    alive_units = sum(unit.is_alive() for unit in units)
                    if alive_units > 0:
                        avg_energy = np.mean([unit.energy for unit in units if unit.is_alive()])
                        avg_money = np.mean([unit.money for unit in units if unit.is_alive()])
                    else:
                        avg_energy = 0.0
                        avg_money = 0.0

                    stats_text = f'Epoch: {epoch + 1}/{NUM_EPOCHS} - Alive Units: {alive_units} - Avg Energy: {avg_energy:.2f} - Avg Money: {avg_money:.2f}'
                    stats_surface = font.render(stats_text, True, (255, 255, 255))

                    screen.blit(stats_surface, (10, 10))
                    pygame.display.flip()

            # save the model
            torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    pygame.display.set_caption('Deep Q-Learning Simulation')
    train_dqn()
    pygame.quit()
