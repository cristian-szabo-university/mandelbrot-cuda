#include "Image.hpp"

Image::Image()
    : width(0), height(0)
{
}

Image::Image(std::int32_t width, std::int32_t height)
    : width(width), height(height)
{
    data.resize(width * height);
}

bool Image::operator==(const Image & other)
{
    return std::equal(data.begin(), data.end(), other.data.begin());
}

std::ostream& operator<<(std::ostream& output, const Image& image)
{
    output << "P6" << std::endl
        << image.width << " "
        << image.height << std::endl
        << 255 << std::endl;

    output.write((char*)&image.data[0], image.data.size() * sizeof(Image::pixel_t));

    return output;
}

std::istream& operator>>(std::istream& input, Image& image)
{
    std::string format;
    input >> format;

    std::string width, height;
    input >> width >> height;

    std::string tmp;
    input >> tmp;

    image.width = std::stoi(width);
    image.height = std::stoi(height);

    image.data.resize(image.width * image.height);

    input.read((char*)image.data.data(), image.data.size() * sizeof(Image::pixel_t));

    return input;
}

bool Image::pixel_t::operator==(const pixel_t & other)
{
    return r == other.r && g == other.g && b == other.b;
}
