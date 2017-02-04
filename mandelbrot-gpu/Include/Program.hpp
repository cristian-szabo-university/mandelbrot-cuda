#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>

#include "docopt.h"
#include "Mandelbrot.hpp"

class Program
{
public:

    Program(std::vector<std::string> args);

    ~Program();

    int run();

private:

    std::map<std::string, docopt::value> args;

    bool save_ppm_file(const std::string& file_name, std::vector<rgb_t> img_data, int width, int height);

};
