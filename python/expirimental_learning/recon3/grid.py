import pygame
from agent import Agent

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def update(self):
        for agent in self.agents:
            if agent.alive:
                agent.update(self)
        self.agents = [agent for agent in self.agents if agent.alive]

    def draw(self, screen, generation, avg_age, max_age, min_age):
        screen.fill((255, 255, 255))
        for agent in self.agents:
            color = (0, 0, 255) if agent.sex == 'M' else (255, 0, 0)
            pygame.draw.circle(screen, color, (agent.x * 12, agent.y * 8), 5)

        font = pygame.font.SysFont(None, 24)
        text = font.render(f'Generation: {generation}, Avg Age: {avg_age:.2f}, Max Age: {max_age}, Min Age: {min_age}', True, (0, 0, 0))
        screen.blit(text, (10, 10))
