
from .status import Status
from DGL.micro import Settings, p
from .action import Action

## Reward Functions for our Agent
def calculateTargetReward(prev_mag, target, x, y):
    '''
    Calculates the reward based on the distance to the target'''
    previous_magnitude = prev_mag
    target_direction_vector, magnitude = p(x, y, target.x, target.y)
    return (magnitude, magnitude - previous_magnitude, target_direction_vector)

def calculateReward(prev_d, x, y, target, action):
    '''
    Used to tell us how rewarding a potential action could be
            '''
    reward = 0

    # Target reward is based on closing the distance to the target
    magnitude, target_reward, target_direction_vector = calculateTargetReward(prev_d, target, x, y)
    reward += target_reward

    if action == Action.Move:
        reward -= 1
    else:
        reward += Settings.LIFETIME_REWARD_SCALAR.value

    # If energy goes up
    if action == Action.Eat:
        reward += Settings.FOOD_REWARD.value

    # If wealth goes up
    # TODO: Test if Status Order changes the simulation outcomes
    if action == Action.Work:
        reward += Settings.WORK_REWARD.value

    # If we reproduce
    if action == Action.Mate:
        reward += Settings.SEX_REWARD.value

    # TODO: Add a randomness for 'happiness factor' to the macro genetic scale 
    #       where some agents care more about different things

    return {
        'magnitude': magnitude,
        'reward': reward,
        'target_direction_vector': target_direction_vector
    }
