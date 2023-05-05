from game import Game
import numpy as np
import random

# implement Reinforcement Learning to learn how to play the space invaders knock off
# use the Q-learning algorithm to learn how to play the game
class QLearningAgent:
    def __init__(self):
        # Q-table (state, action) -> value
        self.learning_rate = 0.1
        # discount factor (how much we care about future rewards)
        self.discount_factor = 0.9
        # epsilon-greedy strategy (how to balance exploration/exploitation)
        self.epsilon = 0.1
        # exploration rate at the beginning
        self.exploration_rate = 1.0
        # how much we reduce the exploration rate every step
        self.exploration_decay_rate = 0.001
        # size 
        self.n_states = None
        # left, right, stop, shoot, do nothing
        self.n_actions = 5

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]

        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return self.get_max_index(self.q_table[state])
        
    def get_max_index(self, array):
        max_value = max(array)
        return array.index(max_value)
    
    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[state][action] = new_q
    
    def update_exploration_rate(self):
        self.exploration_rate *= (1 - self.exploration_decay_rate)


def main():
    game = Game()
    testing = True
    
    # Game Development
    if testing: 
        while game.running:
            game.run()
    # Reinforcement Learning
    # else:
    
if __name__ == "__main__":
    main()