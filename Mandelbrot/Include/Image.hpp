#pragma once

#include <vector>
#include <cstdint>
#include <sstream>
#include <algorithm>
#include <functional>

class Image
{
public:

    struct pixel_t
    {
        std::uint8_t r, g, b;

        bool operator==(const pixel_t& other);
    };

    Image();

    Image(std::int32_t width, std::int32_t height);

    std::vector<pixel_t> data;

    std::int32_t width;

    std::int32_t height;

    bool operator==(const Image& other);

    friend std::ostream &operator<<(std::ostream& output, const Image& other);

    friend std::istream &operator>>(std::istream& input, Image& other);

};
