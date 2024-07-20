import pygame
from pygame_widgets.slider import Slider
from DGL.cosmos import Log, LogLevel, Settings, Color
from DGL.society.agency import State
from .grid import Grid

background = Color.BACKGROUND.value
grid_size = Settings.GRID_SIZE.value
cell_size = Settings.CELL_SIZE.value

class Engine(Grid):
    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_caption('For Real Life!?')
        self.clock = pygame.time.Clock()
        self.running = True
        self.target_selection = None # Starts as an XY and finds a given Cell Type, and passes to update_world

    def selectCell(self, mouse_pos):
        '''
        Gives us the ability to select a cell from the UI'''
        x, y = mouse_pos
        x = x // Settings.CELL_SIZE.value
        y = y // Settings.CELL_SIZE.value

        self.target_selection = (x, y)

        Log(LogLevel.INFO, "World", f"Selected {x}, {y}")

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.selectCell(pygame.mouse.get_pos()) # Move this assignment to the World class     

    def gui(self):
        blit_length = Settings.CELL_SIZE.value * Settings.GRID_SIZE.value // 8
        relative_offset = Settings.GRID_END.value * Settings.CELL_SIZE.value - blit_length * 1.35
        frame_height = Settings.CELL_SIZE.value * Settings.GRID_SIZE.value // 6
        frame_start = 11

        # Transparent Frame
        new_surface = pygame.Surface((blit_length * 2, frame_height))
        new_surface.set_alpha(80)
        new_surface.fill((0, 0, 0))
        self.screen.blit(new_surface, (relative_offset, frame_start))

        # Draw Text
        height_offset = 24
        
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward", 'Selected']
        values = [State.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward, self.target_selection.__class__.__name__]
        font = pygame.font.Font(None, 22)

        for i in range(len(sections)):

            section = sections[i]
            section = font.render(f"{section}:", True, (222, 222, 222, 80))
            self.screen.blit(section, (relative_offset + 22, i * 22 + height_offset))

            value = values[i]
            value = font.render(f"{value}", True, (255, 255, 255, 80))
            width_offset = value.get_width() + 16
            self.screen.blit(value, (grid_size * cell_size - width_offset - 22, i * 22 + height_offset))

    def runLoop(self, world_update):
        '''
        Simply handles pygame, and drawing the cells and gui, and calls the world_update function
        '''        
        Log(LogLevel.INFO, "Ingine", " ~ Running MAIN Engine Loop ~")

        while self.running:
            self.screen.fill(background)
            self.drawCells()
            world_update() # This is coming from the 'World' class, where we are drawing the unites
            self.gui()
            self.events()
            pygame.display.flip()
            self.clock.tick(Settings.FPS.value)
            
        pygame.quit()