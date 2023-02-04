#pragma once
#include "GUI.h"

typedef sf::Color colors;

typedef struct RenderObjects{
	int n_nodes = 0;
	vector<sf::CircleShape> Nodes;
	int n_connections = 0;
	vector<sf::RectangleShape> Connections;
	int n_systems = 0;
	vector<sf::RectangleShape> Systems;
	int n_data = 0;
	vector<sf::CircleShape> Data;
} RenderObjects;

static RenderObjects* Objects(new RenderObjects);

void createNode(float radius, float location_x, float location_y, colors color) {
	sf::CircleShape N1(radius);
	N1.setFillColor(color);
	N1.setOrigin(radius, radius);
	N1.setPosition(location_x, location_y);
	Objects->Nodes.push_back(N1);
	Objects->n_nodes++;
}

void createSystem(float width, float height, float location_x, float location_y, colors color) {
	sf::RectangleShape S(sf::Vector2f(width, height));
	S.setFillColor(color);
	S.setOrigin(width / 2.0f, width / 2.0f);
	S.setPosition(location_x, location_y);
	Objects->Systems.push_back(S);
	Objects->n_systems++;
}