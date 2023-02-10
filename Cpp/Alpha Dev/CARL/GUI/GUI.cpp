#include "GUI.h"
#include "Systems.h"
#include "Objects.h"
#include <map>
#include <stdio.h>

	///////////////////////////
	// Variable Declarations //
	///////////////////////////

typedef sf::Event Action;
static sf::Vector2i prevMousePosition;


	///////////////////////
	// Support Functions //
	///////////////////////


	// Filter Intactions
static inline void queueEvents(Action event, sf::RenderWindow* display) {
	switch (event.type) {
		case Action::MouseWheelMoved: 
		{
			sf::View view{ display->getView() };
			view.zoom((-(fscalar)event.mouseWheel.delta * 0.05f + 1.0f));
			display->setView(view);
			break;
		}
		case Action::MouseButtonPressed: {
			if (event.mouseButton.button == sf::Mouse::Left) {
				// some action
			}
			if (event.mouseButton.button == sf::Mouse::Right) {
				// some menu
			}
			break;
		}
		case Action::MouseMoved: {
			sf::Vector2i newMousePosition = sf::Mouse::getPosition(*display);

			if (sf::Mouse::isButtonPressed(sf::Mouse::Middle)) {
				sf::View view{ display->getView() };
				view.move(sf::Vector2f{ prevMousePosition - newMousePosition });
				display->setView(view);
			}

			prevMousePosition = newMousePosition;
			break;
		}
		case Action::Closed:
		{
			display->close();
			return;
		}
    default:
      return;
	}
}

void drawObjects(sf::RenderWindow* display) {
	for (sf::RectangleShape System : Objects->Systems) {
		display->draw(System);
	}

	for (sf::RectangleShape Line : Objects->Connections) {
		display->draw(Line);
	}

	for (sf::CircleShape Data : Objects->Data) {
		display->draw(Data);
	}

	for (sf::CircleShape Node : Objects->Nodes) {
		display->draw(Node);
	}
}


	// Main Render Loop //
static inline void runRenderLoop(sf::RenderWindow* display) {
	while (display->isOpen()) {
		Action event;

		while (display->pollEvent(event)) {
			queueEvents(event, display);
		}

		display->clear();
		drawObjects(display);
		display->display();
	}
}

void prepareEnvironment() {
	createSystem(&RLSystem);
	createSystem(&GANSystem);
	createSystem(&CNNSystem);
	createSystem(&RNNSystem);
	createSystem(&SNNSystem);
}

	// Entry Point //
void initGUI() {
	sf::RenderWindow display(sf::VideoMode(window_w, window_h), "CARL!");
	sf::View view(sf::FloatRect(0.0f, 0.0f, (fscalar)window_w, (fscalar)window_h));
	display.setView(view);

	prepareEnvironment();

	runRenderLoop(&display);
}
