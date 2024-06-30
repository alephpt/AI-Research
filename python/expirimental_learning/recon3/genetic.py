from agent import Agent
import random

class GeneticAlgorithm:
    def __init__(self):
        self.generation = 0

    def select_parents(self, agents):
        if len(agents) < 2:
            return agents
        return sorted(agents, key=lambda x: x.age, reverse=True)[:50]

    def reproduce(self, parent1, parent2, id):
        x, y = random.randint(0, 100), random.randint(0, 100)
        sex = 'M' if random.random() < 0.5 else 'F'
        child = Agent(id, x, y, sex)
        child.parents = [parent1, parent2]
        parent1.children.append(child)
        parent2.children.append(child)
        parent1.happiness += 25
        parent2.happiness += 25
        return child

    def new_generation(self, agents):
        parents = self.select_parents(agents)
        if len(parents) < 2:
            return [Agent(i, random.randint(0, 100), random.randint(0, 100), 'M' if random.random() < 0.5 else 'F') for i in range(100)]

        new_agents = []
        for i in range(100):
            parent1, parent2 = random.sample(parents, 2)
            new_agents.append(self.reproduce(parent1, parent2, i))
        self.generation += 1
        return new_agents
