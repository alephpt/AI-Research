import pygame
from pygame_widgets.slider import Slider
from DGL.cosmos import Log, LogLevel, Settings
from DGL.society.agency import State
from .grid import Grid

background = Settings.BACKGROUND_COLOR.value
grid_size = Settings.GRID_SIZE.value
cell_size = Settings.CELL_SIZE.value

class Engine(Grid):
    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_caption('For Real Life!?')
        self.clock = pygame.time.Clock()
        self.running = True
        self.selected = None

    def selectCell(self, mouse_pos):
        '''
        Gives us the ability to select a cell from the UI'''
        x, y = mouse_pos
        x = x // Settings.CELL_SIZE.value
        y = y // Settings.CELL_SIZE.value

        self.selected = (x, y)

        Log(LogLevel.INFO, "World", f"Selected {x}, {y}")

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.selectCell(pygame.mouse.get_pos()) # Move this assignment to the World class     

    def draw(self):
        self.screen.fill(background)
        self.drawCells()
        self.gui()

    def gui(self):
        blit_length = Settings.CELL_SIZE.value * 13
        relative_offset = Settings.GRID_END.value * Settings.CELL_SIZE.value - blit_length * 1.35
        frame_height = Settings.CELL_SIZE.value * 19
        frame_start = 11

        # Transparent Frame
        new_surface = pygame.Surface((blit_length * 2, frame_height))
        new_surface.set_alpha(80)
        new_surface.fill((0, 0, 0))
        self.screen.blit(new_surface, (relative_offset, frame_start))

        # Draw Text
        height_offset = 24
        
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward", 'Selected']
        values = [State.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward, self.selected.__class__.__name__]
        font = pygame.font.Font(None, 22)

        for i in range(len(sections)):
            section = sections[i]
            value = values[i]

            section = font.render(f"{section}:", True, (222, 222, 222, 80))
            value = font.render(f"{value}", True, (255, 255, 255, 80))
            width_offset = value.get_width() + 16
            self.screen.blit(section, ((grid_size - cell_size * 2 - 2) * cell_size, i * 22 + height_offset))
            self.screen.blit(value, (grid_size * cell_size - width_offset - 22, i * 22 + height_offset))

    def runLoop(self, callback):
        Log(LogLevel.INFO, "Ingine", " ~ Running MAIN Engine Loop ~")

        while self.running:
            self.events()
            self.update()
            self.draw()
            callback()
            self.updatePopulation(self.selected)
            pygame.display.flip()
            self.clock.tick(Settings.FPS.value)
            
        pygame.quit()