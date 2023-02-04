#pragma once
#include "../Types/Vector.h"
#include <SFML/Graphics.hpp>


typedef enum GuiSystemTypes {
	RL_GUI_SYSTEM,
	GAN_GUI_SYSTEM,
	CNN_GUI_SYSTEM,
	RNN_GUI_SYSTEM,
	SNN_GUI_SYSTEM
} GuiSystemTypes;

void initGUI();



