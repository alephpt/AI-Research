
from .direction import p

## Reward Functions for our Agent
def calculateTargetReward(magnitude, target, x, y):
    '''
    Calculates the reward based on the distance to the target'''
    previous_magnitude = magnitude
    target_direction_vector, magnitude = p(x, y, target.x, target.y)
    return (magnitude, magnitude - previous_magnitude, target_direction_vector)

def calculateReward(target, status):
    reward = 0

    # If the distance towards the correct target is less
    magnitude, target_reward, target_direction_vector = calculateTargetReward(target)
    reward += target_reward

    # If energy goes up
    if status == Status.Eating:
        reward

    # If wealth goes up
    # TODO: Test if Status Order changes the simulation outcomes
    if status == Status.Working:
        reward += 1

    # TODO: Add a randomness for 'happiness factor' to the macro genetic scale 
    #       where some agents care more about different things

    # Happier Lives are better
    reward += self.happiness

    # Longer Lives are better
    reward += self.age

    return {
        'magnitude': magnitude,
        'reward': reward,
        'target_direction_vector': target_direction_vector
    }
