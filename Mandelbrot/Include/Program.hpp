#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>

#include "docopt.h"
#include "Image.hpp"

class Program
{
public:

    Program(std::vector<std::string> args);

    ~Program();

    int run();

private:

    std::map<std::string, docopt::value> args;

    Image pixel_colour;

    std::uint64_t generate_mandelbrot(Image& image, double cx, double cy);

    std::uint64_t generate_mandelbrot_optimised(Image& image, double cx, double cy);

};
