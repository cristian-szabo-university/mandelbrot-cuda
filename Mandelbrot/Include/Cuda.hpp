#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <sstream>
#include <iostream>
#include <numeric>

#include "cuda_occupancy.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "Image.hpp"

namespace cuda
{
    std::uint64_t generate_mandelbrot(Image& image, double cx, double cy);
}
