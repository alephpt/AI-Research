#pragma once
#include "../Types/Types.h"
#include "Systems.h"
#include "GUI.h"

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

void createNode(fscalar radius, fscalar location_x, fscalar location_y, colors color) {
	sf::CircleShape N1(radius);
	N1.setFillColor(color);
	N1.setOrigin(radius, radius);
	N1.setPosition(location_x, location_y);
	Objects->Nodes.push_back(N1);
	Objects->n_nodes++;
}

void createSystem(GUISystem* System) {
	sf::RectangleShape S(System->dimensions);
	S.setFillColor(System->color);
	S.setOrigin(System->dimensions.x / 2.0f, System->dimensions.y / 2.0f);
	S.setPosition(System->region);
	Objects->Systems.push_back(S);
	Objects->n_systems++;
}