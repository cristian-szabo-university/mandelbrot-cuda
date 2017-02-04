#include "Program.hpp"

Program::Program(std::vector<std::string> cli)
{
    args = docopt::docopt(R"(
    Usage:
        mandelbrot generate (<ppm_file>) [--width=<px> --height=<px> --validate_time=<count>]
        mandelbrot (-h | --help)
        mandelbrot --version

    Options:
        -w --width=<px>             Image width.
        -h --height=<px>            Image height.
        -v --validate_time=<count>  Check time.
        --help                      Show this screen.
        --version                   Show version.
    )", cli, true, "MandelbrotCPU 1.0.0");

    pixel_colour =
    {
        {  66,  30,  15 },{  25,   7,  26 },{   9,   1,  47 },{   4,   4,  73 },
        {   0,   7, 100 },{  12,  44, 138 },{  24,  82, 177 },{  57, 125, 209 },
        { 134, 181, 229 },{ 211, 236, 248 },{ 241, 233, 191 },{ 248, 201,  95 },
        { 255, 170,   0 },{ 204, 128,   0 },{ 153,  87,   0 },{ 106,  52,   3 }
    };
}

Program::~Program()
{
    args.clear();
}

int Program::run()
{
    using namespace std::chrono;

    if (!args["generate"].asBool())
    {
        throw std::runtime_error("Command not defined yet!");
    }

    const int width = std::stoi(args["--width"].asString());
    const int height = std::stoi(args["--height"].asString());
    const int checks = std::stoi(args["--validate_time"].asString());
    std::string ppm_file = args["<ppm_file>"].asString();
    const double scale = 1.0 / (width / 4.0);

    std::vector<rgb_t> img_data;
    float min_exec_time = std::numeric_limits<float>::max();

    for (int i = 0; i< checks; i++)
    {
        float elapsed_time = create_image(img_data, width, height, scale);

        if (min_exec_time > elapsed_time)
        {
            min_exec_time = elapsed_time;
        }
    }

    if (!save_ppm_file(ppm_file, img_data, width, height))
    {
        throw std::runtime_error("Failed to save the ppm file!");
    }

    std::cout << "Generate time: " << min_exec_time / 1000.0 << " seconds." << std::endl;

    return 0;
}

float Program::create_image(std::vector<rgb_t>& img_data, const int width, const int height, const double scale)
{
    using namespace std::chrono;

    const double cx = -0.6, cy = 0.0;
    const std::uint8_t max_iter = std::numeric_limits<std::uint8_t>::max();

    img_data.resize(width * height, { 0, 0, 0 });

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < height; i++)
    {
        const double y = (i - height / 2) * scale + cy;

        for (int j = 0; j < width; j++)
        {
            const double x = (j - width / 2) * scale + cx;

            double zx, zy, zx2, zy2;
            std::uint8_t iter = 0;

            zx = hypot(x - 0.25, y);
            if (x < zx - 2 * zx * zx + 0.25) iter = max_iter;
            if ((x + 1)*(x + 1) + y * y < 0.0625) iter = max_iter;

            // f(z) = z^2 + c
            //
            // z = x + i * y;
            // z^2 = x^2 - y^2 + i * 2xy
            // c = x0 + i * y0;
            zx = zy = zx2 = zy2 = 0.0;
            do {
                zy = 2.0 * zx * zy + y; // y = Img(z^2 + c) = 2xy + y0;
                zx = zx2 - zy2 + x; // x = Re(z^2 + c) = x^2 - y^2 + x0;
                zx2 = zx * zx;
                zy2 = zy * zy;
            } while (iter++ < max_iter && zx2 + zy2 < 4.0);

            if (iter < max_iter && iter > 0)
            {
                const std::uint8_t px_idx = iter % 16;

                img_data[i * width + j] = pixel_colour[px_idx];
            }
        }
    }

    auto end_time = high_resolution_clock::now();
    auto elapsed_time = duration_cast<milliseconds>(end_time - start_time);

    return elapsed_time.count();
}

bool Program::save_ppm_file(const std::string& file_name, std::vector<rgb_t> img_data, int width, int height)
{
    std::ofstream file(file_name);

    if (!file.is_open())
    {
        return false;
    }

    file << "P6" << std::endl
        << width << " "
        << height << std::endl
        << 255 << std::endl;

    file.write((char*)img_data.data(), width * height * sizeof(rgb_t));

    file.close();

    return true;
}