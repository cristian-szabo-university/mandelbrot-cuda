#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <sstream>

#include <cuda_occupancy.h>
#include <cuda_runtime.h>

#include "Image.hpp"

namespace cuda
{
    std::uint64_t generate_mandelbrot(Image& image, double cx, double cy);
}
