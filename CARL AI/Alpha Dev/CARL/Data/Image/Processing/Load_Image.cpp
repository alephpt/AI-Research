#include "Load_Image.h"
#ifdef _win32
#include "windows.h"
#endif

Image::Image(std::string image_path) {
    if (!image.loadFromFile(image_path)) 
    {
        printf("Initializing Image Type failed: \nLooking for - %s\n", image_path.c_str());
    }
   
    width = image.getSize().x;
    height = image.getSize().y;
}

Image::~Image() { image_properties.clear(); }

int Image::getWidth() { return width; }
int Image::getHeight() { return height; }

ftensor3d Image::getTensor() { 
    image_properties = ftensor3d(4, fmatrix(height, vector<fscalar>(width)));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            sf::Color p = image.getPixel(x, y);

            image_properties[0][y][x] = p.r / 255.f;
            image_properties[1][y][x] = p.g / 255.f;
            image_properties[2][y][x] = p.b / 255.f;
            image_properties[3][y][x] = ((p.r * 0.299f) + (p.g * 0.587f) + (p.b * .114f)) / 255.f;
        }
    }

    return image_properties; 
}
