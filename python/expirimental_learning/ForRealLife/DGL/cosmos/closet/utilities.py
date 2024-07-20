from ..settings import LogLevel
from .logger import Log
from .vector import Vector
import math

def magnitude(x, y):
    '''
    Returns the relative magnitude of the vector'''
    return math.sqrt(x ** 2 + y ** 2)


def vector(x, y, magnitude):
    '''
    Returns the normalized vector and magnitude'''
    return Vector((x / magnitude, y / magnitude), magnitude)


def calculateDirection(x, y, tX, tY):
    '''
    Calculates the direction vector given the x, y, and target x, y, and returns the vector and magnitude'''
    dX = tX - x
    dY = tY - y

    if dX == 0 and dY == 0:
        return Vector((0, 0), 0)

    return vector(x, y, magnitude(dX, dY))

## Used to calculate the Directional Magnitude of a Vector
def p(x1, y1, x2, y2):
    '''
    Calculate the Actual Directional Magnitude of a Vector'''
    Log(LogLevel.VERBOSE, "[p]", f"Calculating Directional Magnitude of a Vector: xy1:{x1, y1}, xy2:{x2, y2}")
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return Vector.New()
    
    magnitude = (dx ** 2 + dy ** 2) ** 0.5
    dxm = round(dx / magnitude)
    dym = round(dy / magnitude)
    return Vector((dxm, dym), magnitude)


## Reward Functions for our Units Movement towards the Target
def track(prev_d, x, y, target):
    '''
    'findBest' utility function to calculate the reward for the unit

    Returns:
    (distance, cartesian_direction_vector)

    Parameters:
    prev_d: float - The previous distance to the target
    x: int - The x coordinate of the unit
    y: int - The y coordinate of the unit
    target: Cell - The target of the unit
            '''
    # If the previous distance is 0, we are at the target
    if target is None:
        return Vector.Null()

    if prev_d is 0:
        return Vector.New()

    # Otherwise we want to unwrap the target, whether it's a tuple or otherwise
    t_x, t_y = (0, 0)
    if isinstance(target, tuple):
        t_x, t_y = target
    else:
        t_x, t_y = target.xy()

    return p(x, y, t_x, t_y)
