#include "cuda_kernel.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

/**
 * @brief CUDA kernel to apply a Gaussian filter to an image.
 *
 * @param input Pointer to the input image data.
 * @param output Pointer to the output image data.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param kernel Pointer to the Gaussian kernel.
 * @param kernel_size Size of the Gaussian kernel (assumed to be square).
 */
__global__ void GaussianFilter(const unsigned char *input, unsigned char *output, int width, int height, const float *kernel, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_k = kernel_size / 2;

    if (x < width && y < height)
    {
        float sum = 0.0f;
        for (int ky = -half_k; ky <= half_k; ++ky)
        {
            for (int kx = -half_k; kx <= half_k; ++kx)
            {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                sum += input[iy * width + ix] * kernel[(ky + half_k) * kernel_size + (kx + half_k)];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

/**
 * @brief CUDA kernel to convert a color image to grayscale.
 *
 * @param input Pointer to input color image data (RGB format).
 * @param output Pointer to output grayscale image data.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void colorToBW(unsigned char *input, unsigned char *output, size_t width, size_t height)
{
    // Calculate the x and y coordinates of the pixel this thread is responsible for
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the coordinates are within the image bounds
    if (x >= width || y >= height)
        return;

    // Calculate the 1D index of the output pixel
    int idx_out = y * width + x;

    // Calculate the 1D index of the input pixel (each pixel has 3 values: R, G, B)
    int idx_in = y * width * 3 + x * 3;

    // Convert to grayscale using the standard luminance conversion formula
    output[idx_out] = 0.299f * input[idx_in + 2] + 0.587f * input[idx_in + 1] + 0.114f * input[idx_in];
}

/**
 * @brief CUDA kernel to apply the Sobel operator to an image.
 *
 * @param input Pointer to input grayscale image data.
 * @param gradient Pointer to output gradient magnitude data.
 * @param direction Pointer to output gradient direction data.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void SobelOperator(unsigned char *input, float *gradient, float *direction, size_t width, size_t height)
{
    // Calculate the x and y coordinates of the pixel this thread is responsible for
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the coordinates are within the image bounds and not at the image border
    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1)
        return;

    // Calculate the horizontal gradient (Gx) using the Sobel operator
    int Gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)] - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)] - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    // Calculate the vertical gradient (Gy) using the Sobel operator
    int Gy = input[(y - 1) * width + (x - 1)] + 2 * input[(y - 1) * width + x] + input[(y - 1) * width + (x + 1)] - input[(y + 1) * width + (x - 1)] - 2 * input[(y + 1) * width + x] - input[(y + 1) * width + (x + 1)];

    // Calculate the gradient magnitude and direction
    float G = sqrtf(Gx * Gx + Gy * Gy);
    float theta = atan2f(Gy, Gx);

    // Store the results in the output arrays
    gradient[y * width + x] = G;
    direction[y * width + x] = theta;
}

/**
 * @brief CUDA kernel for double threshold hysteresis in edge detection.
 *
 * @param input Pointer to input edge magnitude data.
 * @param output Pointer to output edge data.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param low_thresh Low threshold value.
 * @param high_thresh High threshold value.
 */
__global__ void DoubleThresholdHysteresis(unsigned char *input, unsigned char *output, size_t width, size_t height, unsigned char low_thresh, unsigned char high_thresh)
{
    // Calculate the x and y coordinates of the pixel this thread is responsible for
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the coordinates are within the image bounds
    if (x >= width || y >= height)
        return;

    // Calculate the 1D index of the pixel
    int idx = y * width + x;
    unsigned char value = input[idx];

    // Apply double thresholding
    if (value >= high_thresh)
    {
        output[idx] = 255; // Strong edge
    }
    else if (value < low_thresh)
    {
        output[idx] = 0; // Not an edge
    }
    else
    {
        output[idx] = 128; // Weak edge
    }
}

/**
 * @brief CUDA kernel to track edges in the hysteresis phase of edge detection.
 *
 * @param input Pointer to input edge data.
 * @param output Pointer to output edge data.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void HysteresisTrackEdges(unsigned char *input, unsigned char *output, size_t width, size_t height)
{
    // Calculate the x and y coordinates of the pixel this thread is responsible for
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the coordinates are within the image bounds and not at the image border
    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1)
        return;

    // Calculate the 1D index of the pixel
    int idx = y * width + x;

    // If the pixel is a weak edge, check its neighbors
    if (input[idx] == 128)
    {
        // If any of the neighbors is a strong edge, mark it as a strong edge
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
        // Copy the input value to the output
        output[idx] = input[idx];
    }
}

/**
 * @brief CUDA kernel for non-maximum suppression in edge detection.
 *
 * @param gradient Pointer to input gradient magnitude data.
 * @param direction Pointer to input gradient direction data.
 * @param output Pointer to output non-maximum suppression data.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void NonMaxSuppression(float *gradient, float *direction, unsigned char *output, size_t width, size_t height)
{
    // Calculate the x and y coordinates of the pixel this thread is responsible for
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the coordinates are within the image bounds and not at the image border
    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1)
        return;

    // Get the gradient direction and magnitude for the current pixel
    float angle = direction[y * width + x];
    float mag = gradient[y * width + x];

    float mag1, mag2;

    // Determine the neighboring pixels to compare based on the gradient direction
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

    // Perform non-maximum suppression
    if (mag >= mag1 && mag >= mag2)
        output[y * width + x] = mag;
    else
        output[y * width + x] = 0;
}

cv::Mat1b apply_gaussian_filter(const cv::Mat1b &input, const int kernel_size, const float sigma)
{
    const cv::Size size = input.size();
    const int width = size.width;
    const int height = size.height;

    std::vector<float> h_kernel(kernel_size * kernel_size);

    int half_k = kernel_size / 2;
    float sum = 0.0f;

    // Create Gaussian kernel
    for (int y = -half_k; y <= half_k; ++y)
    {
        for (int x = -half_k; x <= half_k; ++x)
        {
            float value = expf(-(x * x + y * y) / (2 * sigma * sigma));
            h_kernel[(y + half_k) * kernel_size + (x + half_k)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernel_size * kernel_size; ++i)
    {
        h_kernel[i] /= sum;
    }

    unsigned char *d_input;
    unsigned char *d_output;
    float *d_kernel;

    // Allocate memory on the GPU
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));

    // Copy input image and kernel to the GPU
    cudaMemcpy(d_input, input.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the Gaussian filter kernel
    GaussianFilter<<<gridDim, blockDim>>>(d_input, d_output, width, height, d_kernel, kernel_size);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    // Create an output image
    cv::Mat1b output(size);
    // output.setTo(cv::Scalar(0));

    // Copy the result back to the host
    cudaMemcpy(output.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return output;
}

cv::Mat1b convert_color_to_bw(const cv::Mat3b &input)
{
    unsigned char *cuda_in;
    unsigned char *cuda_out;

    const cv::Size size = input.size();

    const int &width = size.width;
    const int &height = size.height;

    size_t n_pixels = size.width * size.height;
    size_t size_color = n_pixels * 3;

    // Allocate memory on the device
    cudaMalloc(&cuda_in, size_color);
    cudaMalloc(&cuda_out, n_pixels);

    // Copy the input image data to the device
    cudaMemcpy(cuda_in, (void *)input.data, size_color, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the colorToBW kernel
    colorToBW<<<numBlocks, threadsPerBlock>>>(cuda_in, cuda_out, size.width, size.height);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Check for any errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    // Create an output image
    cv::Mat1b output(size);
    // output.setTo(cv::Scalar(0));

    // Copy the result back to the host
    cudaMemcpy((void *)output.data, cuda_out, n_pixels, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(cuda_in);
    cudaFree(cuda_out);

    return output;
}

cv::Mat1b canny_edge_detection(const cv::Mat1b &input, unsigned char low_thresh, unsigned char high_thresh)
{
    float *cuda_gradient;
    float *cuda_direction;
    unsigned char *cuda_nms;
    unsigned char *cuda_edges;

    const cv::Size size = input.size();

    const int &width = size.width;
    const int &height = size.height;

    size_t n_pixels = width * height;
    size_t size_gray = n_pixels * sizeof(unsigned char);
    size_t size_float = n_pixels * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&cuda_gradient, size_float);
    cudaMalloc(&cuda_direction, size_float);
    cudaMalloc(&cuda_nms, size_gray);
    cudaMalloc(&cuda_edges, size_gray);

    // Copy the input image data to the device
    cudaMemcpy(cuda_edges, input.data, size_gray, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Apply the Sobel operator to calculate gradient magnitude and direction
    SobelOperator<<<numBlocks, threadsPerBlock>>>(cuda_edges, cuda_gradient, cuda_direction, width, height);
    cudaDeviceSynchronize();

    // Perform non-maximum suppression
    NonMaxSuppression<<<numBlocks, threadsPerBlock>>>(cuda_gradient, cuda_direction, cuda_nms, width, height);
    cudaDeviceSynchronize();

    // Apply double thresholding
    DoubleThresholdHysteresis<<<numBlocks, threadsPerBlock>>>(cuda_nms, cuda_edges, width, height, low_thresh, high_thresh);
    cudaDeviceSynchronize();

    // Track edges by hysteresis
    HysteresisTrackEdges<<<numBlocks, threadsPerBlock>>>(cuda_edges, cuda_nms, width, height);
    cudaDeviceSynchronize();

    // Check for any errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    // Create an output image
    cv::Mat1b output(size);
    // output.setTo(cv::Scalar(0));

    // Copy the result back to the host
    cudaMemcpy(output.data, cuda_nms, size_gray, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(cuda_gradient);
    cudaFree(cuda_direction);
    cudaFree(cuda_nms);
    cudaFree(cuda_edges);

    return output;
}