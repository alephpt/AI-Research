from DGL.meso.agency import State
from DGL.micro import LogLevel, Log

# TODO: Implement Utility functions that allow for the creation of a new generation of agents
# TODO: Integrate the Genome into the Genetic Evolution of the Society
# TODO: Create Batch Functions and Parallelize the Genetic Evolution of the Society
class Genome:
    def __init__(self):
        self.reset()

    # When we Shuffle we want 
    # 15-30% of the Oldest Generations
    # 20-35% Mix of Happiest, Healthiest, Wealthiest and Most Rewarded
    # 50% to be Split
    #    20% is Crossover of the Oldest and Top Performers
    #    30% is Crossover of random Slices of Worst Performers

    def reset(self):
        self.min_age = 0
        self.avg_age = 0
        self.max_age = 0

        self.min_happiness = 0
        self.avg_happiness = 0
        self.max_happiness = 0

        self.min_health = 0
        self.avg_health = 0
        self.max_health = 0

        self.min_wealth = 0
        self.avg_wealth = 0
        self.max_wealth = 0

        self.min_reward = 0
        self.avg_reward = 0
        self.max_reward = 0

        self.n_alive = 0

        self.total_age = 0
        self.total_happiness = 0
        self.total_health = 0
        self.total_wealth = 0
        self.total_reward = 0

        self.n_generations = 0

    def updateStatistics(self):
        if 0 == len([agent for agent in self.agents if agent.state != State.Dead]):
            return
        
        self.reset()

        for agent in self.agents:
            if agent.state == State.Dead:
                continue
            
            self.n_alive += 1
            self.n_generations = max(self.n_generations, agent.generation)
            
            self.total_age += agent.age
            #self.total_happiness += agent.happiness
            self.total_health += agent.energy
            self.total_wealth += agent.wealth
            self.total_reward += agent.reward

            self.min_age = min(self.min_age, agent.age)
            self.avg_age += agent.age
            self.max_age = max(self.max_age, agent.age)

            #self.min_happiness = min(self.min_happiness, agent.happiness)
            #self.max_happiness = max(self.max_happiness, agent.happiness)

            self.min_health = min(self.min_health, agent.energy)
            self.max_health = max(self.max_health, agent.energy)

            self.min_wealth = min(self.min_wealth, agent.wealth)
            self.max_wealth = max(self.max_wealth, agent.wealth)

            self.min_reward = min(self.min_reward, agent.reward)
            self.max_reward = max(self.max_reward, agent.reward)

        if 0 == self.n_alive:
            Log(LogLevel.INFO, "No Agents are Alive")
            return

        self.avg_age = self.total_age // self.n_alive
        self.avg_happiness = 0 #self.total_happiness // self.n_alive
        self.avg_health = self.total_health // self.n_alive
        self.avg_wealth = self.total_wealth // self.n_alive
        self.avg_reward = self.total_reward // self.n_alive