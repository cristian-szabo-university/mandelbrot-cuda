#include "Mandelbrot.hpp"

__constant__ rgb_t pixel_colour[16];

__global__ void mandelbrot(rgb_t* img_data, const int width, const int height, const double scale, const int pixel_num)
{
    const std::uint8_t max_iter = 255;
    const double cx = -0.6, cy = 0.0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= pixel_num)
    {
        return;
    }

    int i = idx / width;
    int j = idx % width;

    const double y = (i - height / 2) * scale + cy;
    const double x = (j - width / 2) * scale + cx;

    double zx, zy, zx2, zy2;
    std::uint8_t iter = 0;

    zx = hypot(x - 0.25, y);
    if (x < zx - 2.0 * zx * zx + 0.25) iter = max_iter;
    if ((x + 1)*(x + 1) + y * y < 0.0625) iter = max_iter;

    // f(z) = z^2 + c
    //
    // z = x + i * y;
    // z^2 = x^2 - y^2 + i * 2xy
    // c = x0 + i * y0;
    zx = zy = zx2 = zy2 = 0;
    do {
        zy = 2.0 * zx * zy + y; // y = Img(z^2 + c) = 2xy + y0;
        zx = zx2 - zy2 + x; // x = Re(z^2 + c) = x^2 - y^2 + x0;
        zx2 = zx * zx;
        zy2 = zy * zy;
    } while (iter++ < max_iter && zx2 + zy2 < 4.0);

    if (iter < max_iter && iter > 0)
    {
        const std::uint8_t px_idx = iter % 16;

        img_data[idx] = pixel_colour[px_idx];
    }
}

std::shared_ptr<Device> Device::inst = std::shared_ptr<Device>();

std::shared_ptr<Device> Device::get_inst()
{
    if (!inst)
    {
        inst = std::shared_ptr<Device>(new Device());
    }

    return inst;
}

float Device::create_image(std::vector<rgb_t>& img_data, const int width, const int height, const double scale)
{
    cudaError_t cudaStatus;

    rgb_t* d_img_data;
    int pixel_num = width * height;
    int img_size = pixel_num * sizeof(rgb_t);

    int pixel_block = 1024;

    cudaStatus = cudaMalloc((void**)&d_img_data, img_size);

    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaMalloc failed!");
    }

    cudaStatus = cudaMemset(d_img_data, 0, img_size);

    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaMemset failed!");
    }

    cudaEvent_t start;

    cudaStatus = cudaEventCreate(&start);

    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaEventCreate failed!");
    }

    cudaEvent_t stop;

    cudaStatus = cudaEventCreate(&stop);

    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaEventCreate failed!");
    }

    cudaStatus = cudaEventRecord(start, 0);

    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaEventRecord failed!");
    }

    mandelbrot<<< (pixel_num + pixel_block - 1) / pixel_block, pixel_block >>>(d_img_data, width, height, scale, pixel_num);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) 
    {
        throw std::runtime_error("mandelbrot kernel failed!");
    }

    cudaStatus = cudaEventRecord(stop, 0);
    
    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaEventRecord failed!");
    }

    cudaStatus = cudaEventSynchronize(stop);
    
    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaEventSynchronize failed!");
    }

    float elapsed_time;

    cudaStatus = cudaEventElapsedTime(&elapsed_time, start, stop);

    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaEventElapsedTime failed!");
    }

    img_data.resize(width * height);

    cudaMemcpy(img_data.data(), d_img_data, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_img_data);

    return elapsed_time;
}

Device::Device()
{
    cudaError_t cudaStatus;

    pixel_mapping =
    {
        {  66,  30,  15 }, {  25,   7,  26 }, {   9,   1,  47 }, {   4,   4,  73 },
        {   0,   7, 100 }, {  12,  44, 138 }, {  24,  82, 177 }, {  57, 125, 209 },
        { 134, 181, 229 }, { 211, 236, 248 }, { 241, 233, 191 }, { 248, 201,  95 },
        { 255, 170,   0 }, { 204, 128,   0 }, { 153,  87,   0 }, { 106,  52,   3 }
    };

    cudaStatus = cudaMemcpyToSymbol(pixel_colour, pixel_mapping.data(), pixel_mapping.size() * sizeof(rgb_t));

    if (cudaStatus != cudaSuccess) 
    {
        throw std::runtime_error("cudaMemcpyToSymbol failed!");
    }
}
