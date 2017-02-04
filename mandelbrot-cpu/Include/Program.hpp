#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>

#include "docopt.h"

struct rgb_t
{
    std::uint8_t r, g, b;
};

class Program
{
public:

    Program(std::vector<std::string> args);

    ~Program();

    int run();

private:

    std::map<std::string, docopt::value> args;

    std::vector<rgb_t> pixel_colour;

    float create_image(std::vector<rgb_t>& img_data, const int width, const int height, const double scale);

    bool save_ppm_file(const std::string& file_name, std::vector<rgb_t> img_data, int width, int height);

};
