from game import Game
import numpy as np
import random, os

# implement Reinforcement Learning to learn how to play the space invaders knock off
# use the Q-learning algorithm to learn how to play the game
class QLearningAgent:
<<<<<<< HEAD
    def __init__(self, n_states):
=======
    def __init__(self):
>>>>>>> 691e020aba1eb2a715de3ddf5e958123e8b1cb76
        # Q-table (state, action) -> value
        self.learning_rate = 0.05
        # discount factor (how much we care about future rewards)
        self.discount_factor = 0.9
        # exploration rate at the beginning
        self.exploration_rate = 0.7
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
<<<<<<< HEAD
        
=======

>>>>>>> 691e020aba1eb2a715de3ddf5e958123e8b1cb76
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
<<<<<<< HEAD
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        
        q_value = self.q_table[state][action]
        q_new = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_new - q_value)
    
    # update exploration rate
    def update_exploration_rate(self, epoch):
        self.exploration_rate = 0.01 + (1.0 - 0.01) * np.exp(-self.exploration_decay_rate * epoch)
    
=======

        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]

        q_value = self.q_table[state][action]
        q_new = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_new - q_value)

    # update exploration rate
    def update_exploration_rate(self, epoch):
        # decrease exploration rate linearly over time
        self.exploration_rate = 0.01 + (1.0 - 0.01) * np.exp(-self.exploration_decay_rate * epoch)

>>>>>>> 691e020aba1eb2a715de3ddf5e958123e8b1cb76

def main():
    game = Game()
    testing = False
<<<<<<< HEAD
    epochs = 100
    
    # Game Development
    if testing: 
=======
    epochs = 1000

    # Game Development
    if testing:
>>>>>>> 691e020aba1eb2a715de3ddf5e958123e8b1cb76
        while game.running:
            game.run()
    # Reinforcement Learning
    else:
        print('Reinforcement Learning Started')
<<<<<<< HEAD
        agent = QLearningAgent(len(game.get_states()))
        
=======
        agent = QLearningAgent()

>>>>>>> 691e020aba1eb2a715de3ddf5e958123e8b1cb76
        # open q table and assign it to agent
        if os.path.exists('table.qtf'):
            with open('table.qtf', 'r') as f:
                print('Loading Q Table')
                agent.q_table = eval(f.read())
                print('Q Table Loaded')
<<<<<<< HEAD
            
        total_reward = 0
        for epoch in range(epochs):
            print('Episode: {}/{} Started'.format(epoch, epochs))
            match_reward = 0
            while game.running:
                print('Episode: {}/{} Level: {}, Score: {}, Reward: {}/{}'.format(epoch, epochs, game.level, game.player.score, match_reward, total_reward))
                game.clock.tick(1200)
                state = game.get_states()
                action = agent.get_action(state)
                reward = game.step(action) // 1000 
                game.draw()
                
=======

        total_reward = 0
        for epoch in range(epochs):

            if epoch % (epochs / 10) == 0:
                print('Episode: {}/{} Started'.format(epoch, epochs))
            match_reward = 0
            while game.running:
                game.clock.tick(1200)
                state = game.get_states()
                action = agent.get_action(state)
                reward = game.step(action) // 1000
                game.draw()

>>>>>>> 691e020aba1eb2a715de3ddf5e958123e8b1cb76
                next_state = game.get_states()
                agent.update_q_table(state, action, reward, next_state)
                agent.update_exploration_rate(epoch)
                match_reward += reward
<<<<<<< HEAD
            total_reward += match_reward
            print('Episode: {}/{} Ended. Level: {}, Score: {}, Reward: {}/{}'.format(epoch, epochs, game.level, game.player.score, match_reward, total_reward))
            game.reset()
            
            # save q table
            with open('table.qtf', 'w') as f:
                f.write(str(agent.q_table))
    
if __name__ == "__main__":
    main()
=======
            if epoch % (epochs / 10) == 0:
                total_reward += match_reward
                print('Episode: {}/{} Ended. Level: {}, Score: {}, Reward: {}/{}'.format(epoch, epochs, game.level, game.player.score, match_reward, total_reward))
            game.reset()

            # save q table
            with open('table.qtf', 'w') as f:
                f.write(str(agent.q_table))

if __name__ == "__main__":
    main()
>>>>>>> 691e020aba1eb2a715de3ddf5e958123e8b1cb76
