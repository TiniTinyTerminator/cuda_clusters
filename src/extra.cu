#include "cuda_kernel.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

// Structure to hold RGB color values, packed to save memory space
struct __attribute__((packed)) color_t {
    char r, g, b; ///< Red, Green, and Blue color channels
};

/**
 * @brief Maps points to an image with a specified color.
 * 
 * @param points Array of points to be mapped.
 * @param image The output image where points will be colored.
 * @param w The width of the image.
 * @param h The height of the image.
 * @param n The number of points.
 * @param c The color to be used for mapping points.
 */
__global__ void MapPoints(const IntPoint_t *points, color_t *image, uint32_t w, uint32_t h, uint32_t n, color_t c) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Check if index is out of bounds
    if(idx >= n)
        return;

    const IntPoint_t &p = points[idx];

    // Check if point is within image bounds
    if(p.x >= w || p.y >= h)
        return;

    // Map point to the image with the specified color
    image[p.x + p.y * w] = c;
}

/**
 * @brief Scales points and maps them to an image with a specified color.
 * 
 * @param points A vector of points to be mapped.
 * @param color The color used for mapping points (in BGR order).
 * @param size The size of the output image.
 * @return cv::Mat3b The output image with mapped points.
 * 
 * @throws std::runtime_error If a CUDA error occurs.
 */
cv::Mat3b scale_and_map_points(const std::vector<IntPoint_t> &points, cv::Scalar color, cv::Size size) {
    IntPoint_t *points_d; ///< Device pointer for points
    color_t *color_image; ///< Device pointer for the color image
    
    // Allocate device memory for points
    cudaMalloc(&points_d, points.size() * sizeof(IntPoint_t));
    
    // Allocate device memory for the color image
    cudaMalloc(&color_image, size.height * size.width * sizeof(color_t));

    // Copy points from host to device
    cudaMemcpy(points_d, points.data(), points.size() * sizeof(IntPoint_t), cudaMemcpyHostToDevice);

    // Initialize the color image with zeros
    cudaMemset(color_image, 0, size.height * size.width * sizeof(color_t));

    // Create a color_t object from cv::Scalar (Note: OpenCV uses BGR order)
    color_t c = {static_cast<char>(color[2]), static_cast<char>(color[1]), static_cast<char>(color[0])};

    // Define CUDA kernel configuration
    dim3 threads_per_block(32);
    dim3 num_blocks((points.size() + threads_per_block.x - 1) / threads_per_block.x);

    // Launch the CUDA kernel
    MapPoints<<<num_blocks, threads_per_block>>>(points_d, color_image, size.width, size.height, points.size(), c);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    // Create the output image
    cv::Mat3b output(size);

    // Copy the color image from device to host
    cudaMemcpy(output.data, color_image, size.height * size.width * sizeof(color_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(points_d);
    cudaFree(color_image);

    return output; ///< Return the output image
}
