#include "Mandelbrot.hpp"

#if _DEBUG
#   define cudaCall(cuda_func, ...) { cudaError_t status = cuda_func(__VA_ARGS__); cudaAssert((status), __FILE__, #cuda_func, __LINE__); }
#else
#   define cudaCall(cuda_func, ...) { cudaError_t status = cuda_func(__VA_ARGS__); }
#endif

inline void cudaAssert(cudaError_t status, const char *file, const char* func, int line)
{
    if (status != cudaSuccess)
    {
        std::stringstream ss;
        ss << "Error: " << cudaGetErrorString(status) << std::endl;
        ss << "Func: " << func << std::endl;
        ss << "File: " << file << std::endl;
        ss << "Line: " << line << std::endl;

        throw std::runtime_error(ss.str());
    }
}

__constant__ rgb_t pixel_colour[16];

__global__ void mandelbrot(rgb_t* img_data, const int width, const int height, const double scale, const int pixel_num)
{
    const std::uint8_t max_iter = 255;
    const double cx = -0.6, cy = 0.0;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int idx = i * width + j;

    if (idx >= pixel_num)
    {
        return;
    }

    const double y = (i - (height >> 1)) * scale + cy;
    const double x = (j - (width >> 1)) * scale + cx;

    double zx, zy, zx2, zy2;
    std::uint8_t iter = 0;

    zx = hypot(x - 0.25, y);
    if (x < zx - 2.0 * zx * zx + 0.25)
    {
        return;
    }
    if ((x + 1)*(x + 1) + y * y < 0.0625)
    {
        return;
    }

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
 
    if (iter > 0 && iter < max_iter )
    {
        const std::uint8_t px_idx = iter % 16;

        img_data[idx] = pixel_colour[px_idx];
    }
}

std::shared_ptr<Mandelbrot> Mandelbrot::inst = std::shared_ptr<Mandelbrot>();

std::shared_ptr<Mandelbrot> Mandelbrot::get_inst()
{
    if (!inst)
    {
        inst = std::shared_ptr<Mandelbrot>(new Mandelbrot());
    }

    return inst;
}

float Mandelbrot::create_image(std::vector<rgb_t>& img_data, int width, int height, double scale)
{
    rgb_t* d_img_data;
    int pixel_num = width * height;
    int img_size = pixel_num * sizeof(rgb_t);

    cudaCall(cudaMalloc, (void**)&d_img_data, img_size);
    cudaCall(cudaMemset, d_img_data, 0, img_size);

    cudaEvent_t start;
    cudaCall(cudaEventCreate, &start);

    cudaEvent_t stop;
    cudaCall(cudaEventCreate, &stop);

    cudaCall(cudaEventRecord, start, 0);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    mandelbrot<<< grid, block >>>(d_img_data, width, height, scale, pixel_num);

    cudaCall(cudaGetLastError);
    cudaCall(cudaEventRecord, stop, 0);
    cudaCall(cudaEventSynchronize, stop);
    
    float elapsed_time;
    cudaCall(cudaEventElapsedTime, &elapsed_time, start, stop);

    img_data.resize(width * height);

    cudaCall(cudaMemcpy, img_data.data(), d_img_data, img_size, cudaMemcpyDeviceToHost);

    cudaCall(cudaFree, d_img_data);

    return elapsed_time;
}

Mandelbrot::Mandelbrot()
{
    std::vector<rgb_t> pixel_mapping =
    {
        {  66,  30,  15 }, {  25,   7,  26 }, {   9,   1,  47 }, {   4,   4,  73 },
        {   0,   7, 100 }, {  12,  44, 138 }, {  24,  82, 177 }, {  57, 125, 209 },
        { 134, 181, 229 }, { 211, 236, 248 }, { 241, 233, 191 }, { 248, 201,  95 },
        { 255, 170,   0 }, { 204, 128,   0 }, { 153,  87,   0 }, { 106,  52,   3 }
    };

    cudaCall(cudaMemcpyToSymbol, pixel_colour, pixel_mapping.data(), pixel_mapping.size() * sizeof(rgb_t));
}
