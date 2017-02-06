#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

struct rgb_t
{
    std::uint8_t r, g, b;
};

class Mandelbrot
{
public:

    static std::shared_ptr<Mandelbrot> get_inst();

    float create_image(std::vector<rgb_t>& img_data, int width, int height, double scale);

private:

    static std::shared_ptr<Mandelbrot> inst;

    Mandelbrot();

};


