#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "helper_cuda.h"
#include "cuda_kernel.h"

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

__global__ void colorToBW(unsigned char *input, unsigned char *output, size_t width, size_t height)
{
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    uint32_t idx_out = y * width + x;
    uint32_t idx_in = y * width * 3 + x * 3;

    // Convert to grayscale using standard luminance conversion formula
    output[idx_out] = 0.299f * input[idx_in] + 0.587f * input[idx_in + 1] + 0.114f * input[idx_in + 2];
}

// Sobel Operator Kernel
__global__ void SobelOperator(unsigned char *input, float *gradient, float *direction, size_t width, size_t height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1)
        return;

    int Gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)] - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)] - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    int Gy = input[(y - 1) * width + (x - 1)] + 2 * input[(y - 1) * width + x] + input[(y - 1) * width + (x + 1)] - input[(y + 1) * width + (x - 1)] - 2 * input[(y + 1) * width + x] - input[(y + 1) * width + (x + 1)];

    float G = sqrtf(Gx * Gx + Gy * Gy);
    float theta = atan2f(Gy, Gx);

    gradient[y * width + x] = G;
    direction[y * width + x] = theta;
}

__global__ void DoubleThresholdHysteresis(unsigned char *input, unsigned char *output, size_t width, size_t height, unsigned char low_thresh, unsigned char high_thresh)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    unsigned char value = input[idx];

    if (value >= high_thresh)
    {
        output[idx] = 255;
    }
    else if (value < low_thresh)
    {
        output[idx] = 0;
    }
    else
    {
        output[idx] = 128; // Weak edge
    }
}

__global__ void HysteresisTrackEdges(unsigned char *input, unsigned char *output, size_t width, size_t height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1)
        return;

    int idx = y * width + x;

    if (input[idx] == 128)
    {
        if (input[(y - 1) * width + (x - 1)] == 255 || input[(y - 1) * width + x] == 255 || input[(y - 1) * width + (x + 1)] == 255 ||
            input[y * width + (x - 1)] == 255 || input[y * width + (x + 1)] == 255 ||
            input[(y + 1) * width + (x - 1)] == 255 || input[(y + 1) * width + x] == 255 || input[(y + 1) * width + (x + 1)] == 255)
        {
            output[idx] = 255;
        }
        else
        {
            output[idx] = 0;
        }
    }
    else
    {
        output[idx] = input[idx];
    }
}

// Non-Maximum Suppression Kernel
__global__ void NonMaxSuppression(float *gradient, float *direction, unsigned char *output, size_t width, size_t height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1)
        return;

    float angle = direction[y * width + x];
    float mag = gradient[y * width + x];

    float mag1, mag2;
    if ((angle > -0.3927 && angle <= 0.3927) || (angle > 2.7489 || angle <= -2.7489))
    {
        // Horizontal
        mag1 = gradient[y * width + x - 1];
        mag2 = gradient[y * width + x + 1];
    }
    else if ((angle > 0.3927 && angle <= 1.1781) || (angle > -2.7489 && angle <= -1.9635))
    {
        // 45 degrees
        mag1 = gradient[(y - 1) * width + (x + 1)];
        mag2 = gradient[(y + 1) * width + (x - 1)];
    }
    else if ((angle > 1.1781 && angle <= 1.9635) || (angle > -1.9635 && angle <= -1.1781))
    {
        // Vertical
        mag1 = gradient[(y - 1) * width + x];
        mag2 = gradient[(y + 1) * width + x];
    }
    else
    {
        // 135 degrees
        mag1 = gradient[(y - 1) * width + (x - 1)];
        mag2 = gradient[(y + 1) * width + (x + 1)];
    }

    if (mag >= mag1 && mag >= mag2)
        output[y * width + x] = mag;
    else
        output[y * width + x] = 0;
}

void convert_color_to_bw(unsigned char *input, unsigned char *output, size_t width, size_t height)
{
    unsigned char *cuda_in;
    unsigned char *cuda_out;

    size_t n_pixels = width * height;
    size_t size_color = n_pixels * 3;

    cudaMalloc(&cuda_in, size_color);
    cudaMalloc(&cuda_out, n_pixels);

    cudaMemcpy(cuda_in, input, size_color, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    colorToBW<<<numBlocks, threadsPerBlock>>>(cuda_in, cuda_out, width, height);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    cudaMemcpy(output, cuda_out, n_pixels, cudaMemcpyDeviceToHost);

    cudaFree(cuda_in);
    cudaFree(cuda_out);
}


void canny_edge_detection(unsigned char *input, unsigned char *output, size_t width, size_t height, unsigned char low_thresh, unsigned char high_thresh)
{
    float *cuda_gradient;
    float *cuda_direction;
    unsigned char *cuda_nms;
    unsigned char *cuda_edges;

    size_t n_pixels = width * height;
    size_t size_gray = n_pixels * sizeof(unsigned char);
    size_t size_float = n_pixels * sizeof(float);

    cudaMalloc(&cuda_gradient, size_float);
    cudaMalloc(&cuda_direction, size_float);
    cudaMalloc(&cuda_nms, size_gray);
    cudaMalloc(&cuda_edges, size_gray);

    cudaMemcpy(cuda_edges, input, size_gray, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    SobelOperator<<<numBlocks, threadsPerBlock>>>(cuda_edges, cuda_gradient, cuda_direction, width, height);
    cudaDeviceSynchronize();

    NonMaxSuppression<<<numBlocks, threadsPerBlock>>>(cuda_gradient, cuda_direction, cuda_nms, width, height);
    cudaDeviceSynchronize();

    DoubleThresholdHysteresis<<<numBlocks, threadsPerBlock>>>(cuda_nms, cuda_edges, width, height, low_thresh, high_thresh);
    cudaDeviceSynchronize();

    HysteresisTrackEdges<<<numBlocks, threadsPerBlock>>>(cuda_edges, cuda_nms, width, height);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    
    cudaMemcpy(output, cuda_nms, size_gray, cudaMemcpyDeviceToHost);

    cudaFree(cuda_gradient);
    cudaFree(cuda_direction);
    cudaFree(cuda_nms);
    cudaFree(cuda_edges);
}
