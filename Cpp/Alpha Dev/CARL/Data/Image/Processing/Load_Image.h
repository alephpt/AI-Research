#pragma once
#include <SFML/Graphics.hpp>
#include "../../../Types/Types.h"
#include <string>

class Image {
public:
    Image(std::string);
    ~Image();

    ftensor3d getTensor();
    int getWidth();
    int getHeight();
private:
    int width = 0;
    int height = 0;
    sf::Image image;
    ftensor3d image_properties;
};