
from DGL.micro import Placement, Status, Target, Settings
from .direction import Direction, directions_map
from .reward import calculateReward
import random


# TODO: Implement a way to 'find the next target'
class Agent(Placement):
    def __init__(self, x, y, unit_type):
        super().__init__(x, y, unit_type)
        self.map_size = Settings.MAP_SIZE.value  
        self.age = 0
        self.happiness = 0
        self.sex = random.choice(['M', 'F'])
        self.energy = Settings.INITIAL_E.value # Should we clamp energy
        self.wealth = Settings.INITIAL_W.value # Should we add 'economic' factors? .. If we add a "legal" requirement, will it figure it out? 
        self.status = Status.Alive
        self.magnitude = 0
        self.reward = 0
        self.chosen_direction = Direction.random()
        self.target = None
        self.target_direction_vector = (0, 0)

    # These two functions form the foundations of our future
    def work(self):
        if self.energy >= 10:
            self.wealth += 10
            self.energy -= 5

    def eat(self):
        if self.wealth >= Settings.COST_OF_GOODS.value:
            self.wealth -= Settings.COST_OF_GOODS.value
            self.energy += 10 # TODO: HUGE TEST - In Isolation, determine if fixed values or random values are better
            self.happiness += 1
    
    # This function forms as the gradle of our generational genetics
    def sex(self):
        self.energy -= 10
        self.happiness += 5 # Integrate a large amount of Chance here

    # Q Table - State space
        # Directional Vector
        # Energy Level
        # Wealth Level


    # TODO: Integrate with Queue Table
    def move(self):
        self.energy -= 1
        dx, dy = directions_map[self.chosen_direction.value]

        if 0 <= self.x + dx <= self.map_size - 1 and 0 <= self.y + dy <= self.map_size - 1:
            self.x += dx
            self.y += dy
  
    # Currently only checks if we are dead or not
    def updateState(self):
        # Update the State Space
        if self.energy < 0:
            print(f"Agent {self} has died")
            self.status = Status.Dead

        # We need to determine how to reward our self for what we are doing

        # Update the Q Table

    # Needs to be as random as possible to explore all possible states
    def chooseRandomAction(self, findTarget):
        self.target = findTarget(Target.random()) # TODO: Move this outside of the function

        if self.target is None:
            self.target = self

        # Action
        # Left, Right, Up, Down
        # Change States
        self.chosen_direction = Direction.random()

    # This update function should way potential opportunities, and pick a course of actions,
    # and then update state, reward, and update the Q Table. # 'Caching' happens on the Epoch level
    def update(self, findTarget):
        if self.status == Status.Dead:
            return
        
        self.age += 1
        

        if self.target is None:
            self.chooseRandomAction(findTarget)

        self.chosen_direction = Direction.random()

        # Choose the best action
        # TODO: Look ahead at the next square based on the Q Table OR Do a Random Walk
        # This has to be before the move to ensure the target exists

            ## Percent of Randomness
            # We have the ability to move in a direction with some randomness
        self.move()
        
        # Calculate Collissions
        # Calculate Rewards
        reward_obj = calculateReward(self.target, self.status)
        self.magnitude = reward_obj['magnitude']
        self.target_direction_vector = reward_obj['target_direction_vector']    # We update this here, only AFTER we move
        self.reward += reward_obj['reward']

        # Longer Lives are better
        self.reward += 1 * Settings.LIFETIME_REWARD_SCALAR.value

        # Update the state space
        self.updateState()