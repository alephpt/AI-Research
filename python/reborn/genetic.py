import pygame
import sys
import random
from grid import Grid
from agent import Agent

screen_size = 800
cells = screen_size // 100

class Genetic:
    def __init__(self, pop_size, gene_length, crossover_rate, mutation_rate, generations, env):
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.env = env
        self.population = self.init_population()
        self.generation = 0

    def init_population(self):
        population = []
        for _ in range(self.pop_size):
            sex = 'male' if random.random() < 0.5 else 'female'
            agent = Agent(self.env.state_space, self.env.action_space, sex)
            population.append(agent)
        return population

    def fitness(self, agent):
        total_reward = 0
        for _ in range(agent.episodes_per_agent):
            state = self.env.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                state, reward, done = self.env.step(action)
                total_reward += reward
        return total_reward

    def select_parents(self):
        fitness_scores = [(agent, self.fitness(agent)) for agent in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        selected = fitness_scores[:self.pop_size // 2]
        return [agent for agent, score in selected]

    def crossover(self, parent1, parent2):
        child_q_table = {}
        for state in parent1.q_table:
            if random.random() < 0.5:
                child_q_table[state] = parent1.q_table[state]
            else:
                child_q_table[state] = parent2.q_table[state]
        child = Agent(self.env.state_space, self.env.action_space, 'unknown')
        child.q_table = child_q_table
        return child

    def mutate(self, agent):
        for state in agent.q_table:
            if random.random() < self.mutation_rate:
                action_index = random.randint(0, len(agent.q_table[state]) - 1)
                agent.q_table[state][action_index] += random.uniform(-1, 1)
        return agent

    def evolve(self):
        new_population = []
        parents = self.select_parents()
        while len(new_population) < self.pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population
        self.generation += 1

    def train(self):
        for _ in range(self.generations):
            for agent in self.population:
                agent.train(self.env)
            self.evolve()
            self.visualize_population()

    def visualize_population(self):
        pygame.init()
        screen = pygame.display.set_mode((screen_size, screen_size))
        font = pygame.font.Font(None, 22)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))

            # Draw grid
            for x in range(0, screen_size, cells):
                for y in range(0, screen_size, cells):
                    rect = pygame.Rect(x, y, cells, cells)
                    pygame.draw.rect(screen, (75, 75, 75), rect, 1)

            # Draw workplaces
            for workplace in self.env.workplaces:
                pygame.draw.rect(screen, (205, 205, 0), (workplace[0] * cells, workplace[1] * cells, cells, cells))

            # Draw food sources
            for food_source in self.env.food_sources:
                pygame.draw.rect(screen, (25, 200, 0), (food_source[0] * cells, food_source[1] * cells, cells, cells))

            # Draw agents
            for agent in self.population:
                color = (255, 0, 0) if agent.sex == 'male' else (0, 0, 255)
                position = agent.get_position()
                pygame.draw.circle(screen, color, (position[0] * cells + cells // 2, position[1] * cells + cells // 2), cells // 2)

            # Display generation and fitness information
            generation_text = font.render(f'Generation: {self.generation}', True, (255, 255, 255))
            screen.blit(generation_text, (10, 10))

            avg_fitness = sum([self.fitness(agent) for agent in self.population]) / len(self.population)
            max_fitness = max([self.fitness(agent) for agent in self.population])
            avg_fitness_text = font.render(f'Average Fitness: {avg_fitness:.2f}', True, (255, 128, 255))
            max_fitness_text = font.render(f'Max Fitness: {max_fitness:.2f}', True, (255, 128, 255))
            screen.blit(avg_fitness_text, (10, 35))
            screen.blit(max_fitness_text, (10, 60))

            pygame.display.flip()

        pygame.quit()
        sys.exit()
