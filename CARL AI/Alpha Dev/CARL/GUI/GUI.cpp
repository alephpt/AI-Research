#include "GUI.h"

typedef struct RenderObjects {
	vector<sf::CircleShape> Circles;
} RenderObjects;

static RenderObjects Objects;
static const int window_w = 800;
static const int window_h = 600;
static float circle_r = 100.0f;

static inline void runRenderLoop(sf::RenderWindow* display) {
	while (display->isOpen()) {
		sf::Event event;

		while (display->pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				display->close();
			}
		}

		display->clear();
		display->draw(Objects.Circles[0]);
		display->display();
	}
}

void initGUI() {
	sf::RenderWindow display(sf::VideoMode(window_w, window_h), "CARL!");

	sf::CircleShape N1(circle_r);
	N1.setFillColor(sf::Color::Blue);
	N1.setOrigin(circle_r, circle_r);
	N1.setPosition((float)window_w / 2.0f, (float)window_h / 2.0f);

	Objects.Circles.push_back(N1);

	runRenderLoop(&display);
}
