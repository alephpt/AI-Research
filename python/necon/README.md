# Description and Overview

## Description

### Overview
This Python project utilizes Pygame for creating a simple game engine and demonstrating basic object-oriented programming concepts. It includes modules for engine management (`engine.py`) and object rendering (`blit.py`), enabling interactive shape manipulation within a graphical window.

Ideally this could grow into a system of agents that are able to 'play' against you.

### engine.py
- **Engine Class:** Initializes Pygame, manages the game window, and handles event processing.
- Provides methods for filling the screen, drawing objects, and running the main game loop.
- Controls game termination upon user input.

### blit.py
- **Shape Class:** Defines geometric shapes with attributes like position, size, rotation, and velocity.
- Implements methods for movement, rotation, acceleration, and drawing on the screen.
- Ensures shape boundaries are checked to prevent out-of-bounds rendering.

### main.py
- **Main Functionality:** Integrates the Engine and Shape classes to create a simple interactive application.
- Allows the player (represented by a triangle) to move, rotate, accelerate, and decelerate using keyboard inputs.
- Displays game graphics and updates continuously based on user interaction.

## Dependencies
- Python 3.x
- Pygame

## Author
- Replace with author information or organization details if applicable.
