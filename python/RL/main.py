from game import Game
import numpy as np
import random, os

# implement Reinforcement Learning to learn how to play the space invaders knock off
# use the Q-learning algorithm to learn how to play the game
class QLearningAgent:
    def __init__(self, n_states):
        # Q-table (state, action) -> value
        self.learning_rate = 0.001
        # discount factor (how much we care about future rewards)
        self.discount_factor = 0.9
        # exploration rate at the beginning
        self.exploration_rate = 1.0
        # how much we reduce the exploration rate every step
        self.exploration_decay_rate = 0.001
        # number of states
        # left, right, stop, shoot
        self.n_actions = 4
        # Q-table (state, action) -> value
        self.q_table = {}

    # get action from Q-table
    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        
        # exploration vs exploitation trade-off (epsilon-greedy)
        if random.random() < self.exploration_rate:
            # explore
            return random.randint(0, self.n_actions - 1)
        else:
            # exploit
            return np.argmax(self.q_table[state])

    # update Q-table
    def update_q_table(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0, 0]
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        
        q_value = self.q_table[state][action]
        q_new = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_new - q_value)
    
    # update exploration rate
    def update_exploration_rate(self, epoch):
        self.exploration_rate = 0.01 + (1.0 - 0.01) * np.exp(-self.exploration_decay_rate * epoch)
    

def main():
    game = Game()
    testing = False
    
    # Game Development
    if testing: 
        while game.running:
            game.run()
    # Reinforcement Learning
    else:
        agent = QLearningAgent(len(game.get_states()))
        
        # open q table and assign it to agent
        if os.path.exists('table.qtf'):
            with open('table.qtf', 'r') as f:
                agent.q_table = eval(f.read())
            
        total_reward = 0
        for epoch in range(1000):
            # save previous q table
            with open('table.qtf', 'w') as f:
                f.write(str(agent.q_table))
            match_reward = 0
            while game.running:
                game.clock.tick(8000)
                state = game.get_states()
                action = agent.get_action(state)
                reward = game.step(action) // 1000 
                next_state = game.get_states()
                agent.update_q_table(state, action, reward, next_state)
                agent.update_exploration_rate(epoch)
                match_reward += reward
            total_reward += match_reward
            print('Episode: {}/{}, Score: {}, Reward: {}/{}'.format(epoch, 1000, game.player.score, match_reward, total_reward))
            game.reset()
    
if __name__ == "__main__":
    main()