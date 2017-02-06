#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>

#include "docopt.h"

#include "Config.hpp"

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

    std::uint64_t create_image(std::vector<rgb_t>& img_data, std::int32_t width, std::int32_t height);

    bool save_ppm_file(const std::string& file_name, std::vector<rgb_t> img_data, int width, int height);

};
