import random

screen_size = 800

class Agent:
    def __init__(self, state_space, action_space, sex):
        self.state_space = state_space
        self.action_space = action_space
        self.sex = sex
        self.q_table = self._init_q_table()
        self.episodes_per_agent = random.randint(5, 15)  # Random starting value
        self.age = 0
        self.lifetime_reward = 0
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.position = (random.randint(0, screen_size), random.randint(0, screen_size))

    def _init_q_table(self):
        q_table = {}
        for state in self.state_space:
            q_table[state] = [0] * len(self.action_space)
        return q_table

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            return self.action_space[self.q_table[state].index(max(self.q_table[state]))]

    def learn(self, current_state, action, reward, next_state):
        action_index = self.action_space.index(action)
        best_next_action = max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[current_state][action_index]
        self.q_table[current_state][action_index] += self.alpha * td_error

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env):
        for _ in range(self.episodes_per_agent):
            state = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                self.age += 1
                self.lifetime_reward += reward
            self.update_epsilon()

    def get_position(self):
        return self.position  # Return the current position of the agent

