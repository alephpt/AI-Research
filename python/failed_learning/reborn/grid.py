
class Grid:
    def __init__(self, grid_size=(100, 100), workplaces=[], food_sources=[]):
        self.grid_size = grid_size
        self.state_space = [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]
        self.action_space = ['up', 'down', 'left', 'right']
        self.workplaces = workplaces
        self.food_sources = food_sources
        self.reset()

    def reset(self):
        self.current_position = (0, 0)
        self.money = 0
        self.energy = 10
        self.age = 0
        self.reproduced = False
        return self.current_position

    def step(self, action):
        x, y = self.current_position
        if action == 'up' and y > 0:
            y -= 1
        elif action == 'down' and y < self.grid_size[1] - 1:
            y += 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < self.grid_size[0] - 1:
            x += 1

        self.current_position = (x, y)
        self.age += 1
        self.energy -= 1
        reward = -0.1

        if self.current_position in self.workplaces:
            self.money += 10
            self.energy -= 5
            reward += 1.0
        if self.current_position in self.food_sources and self.money > 5:
            self.energy += 10
            self.money -= 5
            reward += 1.0

        done = self.energy <= 0 or self.age >= 100
        return self.current_position, reward, done

    def get_state(self):
        return (self.current_position, self.money, self.energy, self.age)
