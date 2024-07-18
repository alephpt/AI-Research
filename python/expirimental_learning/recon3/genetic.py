from unit import Unit
import random

class GeneticAlgorithm:
    def __init__(self):
        self.generation = 0

    def select_parents(self, units):
        if len(units) < 2:
            return units
        return sorted(units, key=lambda x: x.age, reverse=True)[:50]

    def reproduce(self, parent1, parent2, id):
        x, y = random.randint(0, 100), random.randint(0, 100)
        sex = 'M' if random.random() < 0.5 else 'F'
        child = Unit(id, x, y, sex)
        child.parents = [parent1, parent2]
        parent1.children.append(child)
        parent2.children.append(child)
        parent1.happiness += 25
        parent2.happiness += 25
        return child

    def new_generation(self, units):
        parents = self.select_parents(units)
        if len(parents) < 2:
            return [Unit(i, random.randint(0, 100), random.randint(0, 100), 'M' if random.random() < 0.5 else 'F') for i in range(100)]

        new_units = []
        for i in range(100):
            parent1, parent2 = random.sample(parents, 2)
            new_units.append(self.reproduce(parent1, parent2, i))
        self.generation += 1
        return new_units
