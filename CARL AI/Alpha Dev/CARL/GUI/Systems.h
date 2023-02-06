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
    GUISystem(float x, float y, float w, float h, colors c) :
        region({ x, y }), dimensions({ w, h }), color(c) {}
} GUISystem;

GUISystem RLSystem((float)(window_w / 6.25), (float)(window_h / 2.125), 150.0f, 200.0f, colors::Red);
GUISystem GANSystem((float)(window_w / 2.5), (float)(window_h / 5), 200.0f, 150.0f, colors::Green);
GUISystem CNNSystem((float)(window_w / 1.125), (float)(window_h / 2.125), 175.0f, 200.0f, colors::Blue);
GUISystem RNNSystem((float)(window_w / 1.5), (float)(window_h / 2.125), 175.0f, 200.0f, colors::Magenta);
GUISystem SNNSystem((float)(window_w / 2.5), (float)(window_h / 1.25), 200.0f, 200.0f, colors::White);

const GUISystem* GUISystems[] { &RLSystem, &GANSystem, &CNNSystem, &RNNSystem, &SNNSystem };