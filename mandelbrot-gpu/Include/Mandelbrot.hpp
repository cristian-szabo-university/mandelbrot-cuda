#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <sstream>

#include <cuda_occupancy.h>
#include <cuda_runtime.h>

struct rgb_t
{
    std::uint8_t r, g, b;
};

class Mandelbrot
{
public:

    static std::shared_ptr<Mandelbrot> get_inst();

    float create_image(std::vector<rgb_t>& img_data, int width, int height);

private:

    static std::shared_ptr<Mandelbrot> inst;

    Mandelbrot();

};


