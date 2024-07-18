import pygame
from color import Color

window_size = 1200
dimensions = 12
gap_size = window_size // dimensions
half_gap = gap_size // 2

offset_rowcount = dimensions + 1

half_dimensions = dimensions // 2
total_index_count = dimensions * dimensions + half_dimensions

global real_y
global y_counter
y_counter = 0
real_y = 0

class Point:
    def __init__(self, idx):
        global real_y, y_counter
        self.y = real_y
        y_counter += 1

        if (real_y % 2) == 0:
            if y_counter % dimensions == 0:
                real_y += 1
                y_counter = 0
            elif y_counter % offset_rowcount == 0:
                real_y += 1
                y_counter = 0


        offset = 0 if idx % 2 == 0 else half_gap # We have to offset for every other row

        self.x = 0
        column_count = 0
        if offset == 0: # If we are offset, we have to +2 cols
            column_count = dimensions
        else:
            column_count = dimensions + 1
        self.x = (idx % column_count) * gap_size - offset
        hue = Color.getHue(self.y / window_size)
        rgb = Color.getColor(self.x, window_size)
        self.color = Color.combine(hue, rgb)
    
    def draw(self, window):
        color = self.color if isinstance(self.color, tuple) else self.color.value
        print("I have a color:", color)
        pygame.draw.circle(window, color, (self.x, self.y), 5)

class Engine:
    def __init__(self):
        self.window = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()
        self.running = True
        self.grid = [Point(i) for i in range(total_index_count)]

    def eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False

    def drawGrid(self):
        for point in self.grid:
            point.draw(self.window)

    def run(self):
        while self.running:
            self.window.fill(Color.BACKGROUND.value)
            self.eventHandler()
            self.drawGrid()
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    print("Creating GeoDesic Pattern")
    Engine().run()


