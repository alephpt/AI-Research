#pragma once
#include "../../Types/Types.h"
#include <SFML/Graphics.hpp>
#include <string>

class Image {
public:
    Image(std::string);
    ~Image();

    tensorf3d getTensor();
    int getWidth();
    int getHeight();
private:
    int width;
    int height;
    sf::Image image;
    tensorf3d image_properties;
};