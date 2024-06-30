import random

class Agent:
    def __init__(self, id, x, y, sex):
        self.id = id
        self.x = x
        self.y = y
        self.sex = sex
        self.energy = 50
        self.money = 50
        self.happiness = 50
        self.age = 0
        self.alive = True
        self.children = []
        self.parents = []

    def move(self, grid):
        self.x = (self.x + random.randint(-1, 1)) % grid.width
        self.y = (self.y + random.randint(-1, 1)) % grid.height

    def eat(self):
        if self.money >= 5:
            self.energy += 10
            self.money -= 5
            self.happiness += 5

    def work(self):
        if self.energy >= 5:
            self.energy -= 5
            self.money += 10
            self.happiness -= 5

    def mate(self, other):
        if self.energy >= 10 and other.energy >= 10:
            self.energy -= 10
            other.energy -= 10
            self.happiness += 20
            other.happiness += 20
            return True
        return False

    def update(self, grid):
        self.age += 1
        self.move(grid)
        self.eat()
        self.work()
        if self.energy <= 0 or self.age >= 100:
            self.alive = False
