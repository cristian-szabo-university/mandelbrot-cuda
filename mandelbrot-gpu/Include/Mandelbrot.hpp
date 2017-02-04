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

class Device
{
public:

    static std::shared_ptr<Device> get_inst();

    float create_image(std::vector<rgb_t>& img_data, const int width, const int height, const double scale);

private:

    static std::shared_ptr<Device> inst;

    std::vector<rgb_t> pixel_mapping;

    Device();

};


