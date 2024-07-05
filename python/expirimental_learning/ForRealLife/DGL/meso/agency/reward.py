
from DGL.micro import p

## Reward Functions for our Agent
def calculateTargetReward(previous_magnitude, target, x, y):
    '''
    Calculates the reward based on the distance to the target'''
    if previous_magnitude is 0:
        target_direction_vector, magnitude = p(x, y, target.x, target.y)
        return (magnitude, magnitude, target_direction_vector)

    target_direction_vector, magnitude = p(x, y, target.x, target.y)
    return (magnitude, magnitude - previous_magnitude, target_direction_vector)

def calculateReward(prev_d, x, y, target):
    '''
    'findBest' utility function to calculate the reward for the agent

    Parameters:
    prev_d: float - The previous distance to the target
    x: int - The x coordinate of the agent
    y: int - The y coordinate of the agent
    target: Unit - The target of the agent
            '''
    reward = 0

    # Target reward is based on closing the distance to the target
    magnitude, target_reward, target_direction_vector = calculateTargetReward(prev_d, target, x, y)
    reward += target_reward

    return {
        'magnitude': magnitude,
        'reward': reward,
        'target_direction_vector': target_direction_vector
    }
