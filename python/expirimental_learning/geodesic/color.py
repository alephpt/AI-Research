from enum import Enum

class Color(Enum):
    BACKGROUND = (16, 16, 16)
    BLACK = (48, 38, 58)
    WHITE = (225, 225, 225)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    @staticmethod
    def new(t):
        return Color(t)

    @staticmethod
    def lerp(color1, color2, t):
        if isinstance(color1, Color):
            color1 = color1.value
        if isinstance(color2, Color):
            color2 = color2.value

        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))
    
    @staticmethod
    def getColor(t, p_n):
        '''Gets the color based on the lerp value for the potential numbers, starting with Red and ending with Green'''
        offset = t / p_n # This gives us what percent of the potential number we are at
        R_threshold = p_n / 3
        G_threshold = p_n - R_threshold

        if offset < R_threshold:
            return Color.lerp(Color.BLACK, Color.RED, offset)
        elif offset < G_threshold:
            return Color.lerp(Color.RED, Color.BLUE, offset)
        else:
            return Color.lerp(Color.BLUE, Color.GREEN, offset)
    
    @staticmethod
    def getHue(t):
        return Color.lerp(Color.BLACK, Color.WHITE, t ** 2)

    def combine(self, other):
        return Color.lerp(self, other, 0.5)