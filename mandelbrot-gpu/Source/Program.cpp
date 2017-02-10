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
    )", cli, true, "MandelbrotGPU 1.0.0");
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

    std::shared_ptr<Mandelbrot> service = Mandelbrot::get_inst();
    std::vector<rgb_t> img_data;
    float min_exec_time = std::numeric_limits<float>::max();

    for (int i = 0; i< checks; i++)
    {
        float elapsed_time = service->create_image(img_data, width, height);

        if (min_exec_time > elapsed_time)
        {
            min_exec_time = elapsed_time;
        }
    }

    if (!save_ppm_file(ppm_file, img_data, width, height))
    {
        throw std::runtime_error("Failed to save the ppm file!");
    }

    std::cout << "Generate time: " << min_exec_time << " ms." << std::endl;

    return 0;
}

bool Program::save_ppm_file(const std::string & file_name, std::vector<rgb_t> img_data, int width, int height)
{   
    std::ofstream file(file_name);

    if (!file.is_open())
    {
        return false;
    }

    file << "P6"    << std::endl
         << width   << " "
         << height  << std::endl
         << 255     << std::endl;

    file.write((char*)img_data.data(), width * height * sizeof(rgb_t));

    file.close();
    
    return true;
}
