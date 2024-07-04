import math

def magnitude(x, y):
    '''
    Returns the magnitude of the vector'''
    return math.sqrt(x ** 2 + y ** 2)


def vector(x, y, magnitude):
    '''
    Returns the normalized vector and magnitude'''
    return (x / magnitude, y / magnitude), magnitude


def calculateDirection(x, y, tX, tY):
    '''
    Calculates the direction vector given the x, y, and target x, y, and returns the vector and magnitude'''
    dX = tX - x
    dY = tY - y

    if dX == 0 and dY == 0:
        return (0, 0), 0

    return vector(x, y, magnitude(dX, dY))

def p(x1, y1, x2, y2):
    '''
    Calculate the Directional Magnitude of a Vector'''
    print(f"Calculating Directional Magnitude of a Vector: {x1, y1, x2, y2}")
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return (0, 0), 0
    
    magnitude = (dx ** 2 + dy ** 2) ** 0.5
    return (dx / magnitude, dy / magnitude), magnitude
