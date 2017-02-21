#include "Program.hpp"

#include "Cuda.hpp"

Program::Program(std::vector<std::string> cli)
{
    args = docopt::docopt(R"(
    Usage:
        mandelbrot generate (gpu|cpu) (<ppm_file>) [<width> <height> --fast]
        mandelbrot compare (<first_file> <second_file>)
        mandelbrot benchmark (gpu|cpu) [<width> <height> --fast]
        mandelbrot (-h | --help)
        mandelbrot --version

    Options:
        --help      Show this screen.
        --version   Show version.
    )", cli, true, "Mandelbrot 1.0.0");

    pixel_colour.data =
    {
        { 66,30,15 },{ 25,7,26 },{ 9,1,47 },{ 4,4,73 },
        { 0,7,100 },{ 12,44,138 },{ 24,82,177 },{ 57,125,209 },
        { 134,181,229 },{ 211,236,248 },{ 241,233,191 },{ 248,201,95 },
        { 255,170,0 },{ 204,128,0 },{ 153,87,0 },{ 106,52,3 }
    };
}

Program::~Program()
{
}

int Program::run()
{
    const double cx = -0.6, cy = 0.0;

    if (args["generate"].asBool())
    {
        std::string ppm_file = args["<ppm_file>"].asString();
        int width = args["<width>"].isString() ? std::stoi(args["<width>"].asString()) : 4096;
        int height = args["<height>"].isString() ? std::stoi(args["<height>"].asString()) : 4096;
       
        Image result(width, height);
        std::uint64_t elapsed_time;

        if (args["cpu"].asBool())
        {
            if (args["--fast"].asBool())
            {
                elapsed_time = generate_mandelbrot_optimised(result, cx, cy);
            }
            else
            {
                elapsed_time = generate_mandelbrot(result, cx, cy);
            }
        }
        else if (args["gpu"].asBool())
        {
            elapsed_time = cuda::generate_mandelbrot(result, cx, cy);
        }

        std::ofstream file(ppm_file);
        file << result;
        file.close();

        std::cout << "Generate time: " << elapsed_time << " ms." << std::endl;
    }
    else if (args["compare"].asBool())
    {
        std::string first_filename = args["<first_file>"].asString();
        std::string second_filename = args["<second_file>"].asString();

        std::ifstream first_file(first_filename, std::ios::binary);
        if (!first_file.is_open())
        {
            throw std::runtime_error("Image " + first_filename + " not found!");
        }

        Image first_image;
        first_file >> first_image;
      
        std::ifstream second_file(second_filename, std::ios::binary);
        if (!second_file.is_open())
        {
            throw std::runtime_error("Image " + second_filename + " not found!");
        }

        Image second_image;
        second_file >> second_image;

        if (first_image == second_image)
        {
            std::cout << "Images are equal!" << std::endl;
        }
        else
        {
            std::cout << "Images are different!" << std::endl;
        }
    }
    else if (args["benchmark"].asBool())
    {
        int width = args["<width>"].isString() ? std::stoi(args["<width>"].asString()) : 4096;
        int height = args["<height>"].isString() ? std::stoi(args["<height>"].asString()) : 4096;

        while (width > 256 && height > 256)
        {
            Image result(width, height);
            std::uint64_t elapsed_time;

            if (args["cpu"].asBool())
            {
                if (args["--fast"].asBool())
                {
                    elapsed_time = generate_mandelbrot_optimised(result, cx, cy);
                }
                else
                {
                    elapsed_time = generate_mandelbrot(result, cx, cy);
                }
            }
            else if (args["gpu"].asBool())
            {
                elapsed_time = cuda::generate_mandelbrot(result, cx, cy);
            }

            std::cout << width << "x" << height << " - " << elapsed_time << std::endl;

            width >>= 1;
            height >>= 1;
        }
    }
    else
    {
        throw std::runtime_error("Command not found!");
    }

    return 0;
}

std::uint64_t Program::generate_mandelbrot(Image& image, double cx, double cy)
{
    std::uint8_t iter_max = std::numeric_limits<std::uint8_t>::max();
    double scale = 1.0 / (image.width / 4.0);

    std::vector<Image::pixel_t*> image_rows(image.height);
    image_rows[0] = &image.data[0];
    for (std::int32_t i = 1; i < image.height; i++)
    {
        image_rows[i] = image_rows[i - 1] + image.width;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (std::int32_t i = 0; i < image.height; i++)
    {
        const double y = (i - image.height / 2) * scale + cy;

        Image::pixel_t* pixel = image_rows[i];

        for (std::int32_t j = 0; j < image.width; j++, pixel++)
        {
            const double x = (j - image.width / 2) * scale + cx;

            double zx, zy, zx2, zy2;
            std::uint8_t iter = 0;

            zx = hypot(x - 0.25, y);

            if (x < zx - 2 * zx * zx + 0.25) iter = iter_max;
            if ((x + 1) * (x + 1) + y * y < 0.0625) iter = iter_max;

            zx = zy = zx2 = zy2 = 0;

            do {
                zy = 2 * zx * zy + y;
                zx = zx2 - zy2 + x;
                zx2 = zx * zx;
                zy2 = zy * zy;
            } while (iter++ < iter_max && zx2 + zy2 < 4);

            *pixel = { iter };
        }
    }

    for (std::int32_t i = 0; i < image.height; i++)
    {
        Image::pixel_t* pixel = image_rows[i];

        for (std::int32_t j = 0; j < image.width; j++, pixel++)
        {
            if (pixel->r > 0 && pixel->r < iter_max)
            {
                const std::uint8_t px_idx = pixel->r % 16;

                *pixel = pixel_colour.data[px_idx];
            }
            else
            {
                *pixel = { 0 };
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

std::uint64_t Program::generate_mandelbrot_optimised(Image & image, double cx, double cy)
{
    std::uint8_t iter_max = std::numeric_limits<std::uint8_t>::max();
    std::int32_t half_width = image.width >> 1;
    std::int32_t half_height = image.height >> 1;
    double scale = 1.0 / (image.width / 4.0);
    __m128d vec_half_width = _mm_set1_pd((double)half_width);
    __m128d vec_scale = _mm_set1_pd(scale);
    __m128d vec_cx = _mm_set1_pd(cx);

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::int32_t i = 0; i < image.height; i++)
    {
        __m128d vec_y = _mm_set1_pd((i - half_height) * scale + cy);

        Image::pixel_t* row = &image.data[i * image.width];

        for (std::int32_t j = 0; j < image.width; j += 2)
        {
            __m128d vec_x = _mm_setr_pd((double)j, (double)(j + 1));

            vec_x = _mm_sub_pd(vec_x, vec_half_width);
            vec_x = _mm_mul_pd(vec_x, vec_scale);
            vec_x = _mm_add_pd(vec_x, vec_cx);

            std::uint8_t iter_count = 0;
            __m128d iter_values = _mm_set1_pd(1.0);
            __m128d iters = _mm_set1_pd(0.0);
            __m128d mask = _mm_set1_pd(1.0);
            __m128d zx = _mm_set1_pd(0.0);
            __m128d zy = _mm_set1_pd(0.0);
            __m128d zx2 = _mm_set1_pd(0.0);
            __m128d zy2 = _mm_set1_pd(0.0);

            do {
                iters = _mm_add_pd(iters, _mm_and_pd(iter_values, mask));
                zy = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(zx, zy), _mm_set1_pd(2.0)), vec_y);
                zx = _mm_add_pd(_mm_sub_pd(zx2, zy2), vec_x);
                zx2 = _mm_mul_pd(zx, zx);
                zy2 = _mm_mul_pd(zy, zy);
            } while (iter_count++ < iter_max && 
                _mm_movemask_pd(iter_values = _mm_cmplt_pd(_mm_add_pd(zx2, zy2), _mm_set1_pd(4.0))) != 0);

            for (int k = 0; k < 2; k++)
            {
                const std::uint8_t iter = (int)iters.m128d_f64[k];

                if (iter > 0 && iter < iter_max)
                {
                    const std::uint8_t idx = iter % 16;

                    row[j + k] = pixel_colour.data[idx];
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}
