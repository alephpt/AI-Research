#include "GUI.h"
#include <map>
#include <stdio.h>



typedef struct {
	vector<sf::CircleShape> Circles;
} RenderObjects;

	// Local Variables
typedef sf::Event Action;
static const int window_w = 800;
static const int window_h = 600;
float centerX = (float)(window_w / 2);
float centerY = (float)(window_h / 2);
static float circle_r = 100.0f;
sf::Vector2i prevMousePosition;
static RenderObjects* Objects(new RenderObjects);


	///////////////////////
	// Support Functions //
	///////////////////////


	// Filter Intactions
static inline void queueEvents(Action event, sf::RenderWindow* display) {
	switch (event.type) {
		case Action::MouseWheelMoved: 
		{
			sf::View view{ display->getView() };
			view.zoom((-(float)event.mouseWheel.delta * 0.05f + 1.0f));
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
				view.move(sf::Vector2f{ newMousePosition - prevMousePosition });
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
		display->draw(Objects->Circles[0]);
		display->display();
	}
}

	// Entry Point //
void initGUI() {
	sf::RenderWindow display(sf::VideoMode(window_w, window_h), "CARL!");

	sf::CircleShape N1(circle_r);
	N1.setFillColor(sf::Color::Blue);
	N1.setOrigin(circle_r, circle_r);
	N1.setPosition(centerX, centerY);
	Objects->Circles.push_back(N1);

	sf::View view(sf::FloatRect(0.0f, 0.0f, (float)window_w, (float)window_h));
	display.setView(view);

	runRenderLoop(&display);
}
