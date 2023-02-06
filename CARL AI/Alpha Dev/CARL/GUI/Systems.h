#pragma once
#include <map>
#include "GUI.h"

typedef enum {
	RL_GUI_SYSTEM,
	GAN_GUI_SYSTEM,
	CNN_GUI_SYSTEM,
	RNN_GUI_SYSTEM,
	SNN_GUI_SYSTEM
} GuiSystemTypes;

typedef struct GUISystem {
    sf::Vector2f region{ 0.0f, 0.0f };
    sf::Vector2f dimensions{ 0.0f, 0.0f};
    colors color = colors::Cyan;
} GUISystem;

GUISystem RLSystem = {
    .region{(float)(window_w / 6.25), (float)(window_h / 2.125)},
    .dimensions{150.0f, 200.0f},
    .color = colors::Red
};

GUISystem GANSystem = {
    .region{(float)(window_w / 2.5), (float)(window_h / 5)},
    .dimensions{200.0f, 150.0f},
    .color = colors::Green
};

GUISystem CNNSystem = {
    .region{(float)(window_w / 1.125), (float)(window_h / 2.125)},
    .dimensions{150.0f, 200.0f},
    .color = colors::Blue
};

GUISystem RNNSystem = {
    .region{(float)(window_w / 1.5), (float)(window_h / 2.125)},
    .dimensions{150.0f, 200.0f},
    .color = colors::Magenta
};

GUISystem SNNSystem = {
    .region{(float)(window_w / 2.5), (float)(window_h / 1.25)},
    .dimensions{200.0f, 200.0f},
    .color = colors::White
};

const GUISystem* GUISystems[] { &RLSystem, &GANSystem, &CNNSystem, &RNNSystem, &SNNSystem };