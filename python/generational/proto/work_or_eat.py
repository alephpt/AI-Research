import concurrent.futures
import numpy as np

# Define initial parameters
INIT_ENERGY = 50
INIT_MONEY = 50
INIT_REWARD = 0
POPULATION = 100
Q_TABLE = lambda: np.zeros((2, 2))
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EXPLORATION_PROB = 0.1
LIFETIME = 1000
EPOCHS = 100

# Define action mapping
actions_map = {0: "Eat", 1: "Work"}

def initializePopulation():
    return [Q_TABLE() for _ in range(POPULATION)]

def selectTopAgents(population, top):
    return sorted(population, key=lambda x: np.sum(x), reverse=True)[:top]

population = initializePopulation()

for epoch in range(EPOCHS):
    print(f"Generation {epoch + 1}")
    # Simulation loop
    for agent in population:
        energy = INIT_ENERGY
        money = INIT_MONEY
        q_table = agent
        total_reward = 0
        
        for year in range(LIFETIME):
            state = np.random.choice([0, 1])
            
            # Choose an action based on epsilon-greedy strategy
            if np.random.uniform(0, 1) < EXPLORATION_PROB:
                action = np.random.choice([0, 1])  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit

            # Perform the chosen action and observe the new state
            chosen_action = actions_map[action]

            if action == 0:  # Eat
                energy += 7
                money -= 10
            else:  # Work
                energy -= 10
                money += 7

            energy = max(0, energy)  # Energy cannot be negative
            money = max(0, money)  # Money cannot be negative
            
            if energy < 5:
                print("Agent died of hunger..")
                break
            
            if money < 5:
                print("Agent died of poverty..")
                break

            # Calculate immediate reward
            reward = energy + money
            total_reward += reward + year

            # Choose the next state
            next_state = np.random.choice([0, 1])

            # Update Q-table
            q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + \
                                    LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[next_state, :]))

        top_agents = selectTopAgents(population, 10)
        new_population = [np.copy(agent) for agent in top_agents]
        new_population.extend([np.copy(agent) for agent in population])
        
        for agent in new_population:
            agent += np.random.normal(0, 0.1, agent.shape) # Add noise to the weights
            
        population = new_population
        
        print(f"Total reward: {total_reward}")

# After training, test the learned policy
state = 0  # Start in the hungry state with no money
best_agent = selectTopAgents(population, 1)[0]

for _ in range(10):
    action = np.argmax(q_table[state, state])
    chosen_action = actions_map[action]
    print(f"Test: {chosen_action}")

    if action == 0:
        energy += 10
        money -= 5
    else:
        energy -= 5
        money += 10
    state = np.random.choice([0, 1])
