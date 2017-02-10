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

struct kernel_t
{
    int width, height;
    int half_width, half_height;
    float scale;

    kernel_t(const int width, const int height) : width(width), height(height)
    {
        half_width = width >> 1;
        half_height = height >> 1;
        scale = 1.0f / (width / 4.0f);
    }
};

__constant__ rgb_t pixel_colour[16] = 
{
    { 66,  30,  15 },{ 25,   7,  26 },{ 9,   1,  47 },{ 4,   4,  73 },
    { 0,   7, 100 },{ 12,  44, 138 },{ 24,  82, 177 },{ 57, 125, 209 },
    { 134, 181, 229 },{ 211, 236, 248 },{ 241, 233, 191 },{ 248, 201,  95 },
    { 255, 170,   0 },{ 204, 128,   0 },{ 153,  87,   0 },{ 106,  52,   3 }
};

__global__ void mandelbrot(const kernel_t info_data, rgb_t* img_data)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= info_data.width || j >= info_data.height)
    {
        return;
    }

    const std::uint8_t max_iter = 255;
    const float cx = -0.6f, cy = 0.0f;
    const float y = (i - info_data.half_width) * info_data.scale + cy;
    const float x = (j - info_data.half_height) * info_data.scale + cx;

    float zx, zy, zx2, zy2;
    
    zx = hypot(x - 0.25f, y);

    if (x < zx - 2.0f * zx * zx + 0.25f || (x + 1) * (x + 1) + y * y < 0.0625f)
    {
        return;
    }

    std::uint8_t iter = 0;
    zx = zy = zx2 = zy2 = 0.0f;

    do {
        zy = 2.0f * zx * zy + y;
        zx = zx2 - zy2 + x;
        zx2 = zx * zx;
        zy2 = zy * zy;
    } while (iter++ < max_iter && zx2 + zy2 < 4.0f);

    if (iter > 0 && iter < max_iter )
    {
        img_data[i * info_data.width + j] = pixel_colour[iter % 16];
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

float Mandelbrot::create_image(std::vector<rgb_t>& img_data, int width, int height)
{
    rgb_t* d_img_data;
    const int img_size = width * height * sizeof(rgb_t);

    cudaCall(cudaMalloc, (void**)&d_img_data, img_size);
    cudaCall(cudaMemset, d_img_data, 0, img_size);

    cudaEvent_t start;
    cudaCall(cudaEventCreate, &start);

    cudaEvent_t stop;
    cudaCall(cudaEventCreate, &stop);

    kernel_t info_data(width, height);

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mandelbrot, 0, 0);
    blockSize = pow(2, floor(log(sqrt(blockSize)) / log(2)));

    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    grid.x = grid.x > minGridSize ? grid.x : minGridSize;
    grid.y = grid.y > minGridSize ? grid.y : minGridSize;

    cudaCall(cudaEventRecord, start, 0);

    mandelbrot<<< grid, block >>>(info_data, d_img_data);

    cudaCall(cudaGetLastError);
    cudaCall(cudaEventRecord, stop, 0);
    cudaCall(cudaEventSynchronize, stop);
    
    float elapsed_time;
    cudaCall(cudaEventElapsedTime, &elapsed_time, start, stop);

    img_data.resize(width * height);

    cudaCall(cudaMemcpy, img_data.data(), d_img_data, img_size, cudaMemcpyDeviceToHost);

    cudaCall(cudaFree, d_img_data);
    cudaCall(cudaEventDestroy, start);
    cudaCall(cudaEventDestroy, stop);

    return elapsed_time;
}

Mandelbrot::Mandelbrot()
{
}
