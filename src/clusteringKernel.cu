#include "cuda_kernel.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

/**
 * @brief A struct to hold boundary pixels.
 */
struct BoundaryPixel
{
  int label;        // The label of the connected component
  IntPoint_t point; // The coordinates of the boundary pixel
};

/**
 * @brief Device-compatible hypot function.
 *
 * @param x The x coordinate.
 * @param y The y coordinate.
 * @return The hypotenuse.
 */
__device__ double DeviceHypot(double x, double y)
{
  return sqrtf(x * x + y * y);
}

/**
 * @brief CUDA kernel to initialize labels for connected component labeling.
 *
 * @param input Pointer to input binary image data.
 * @param labels Pointer to output labels data.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void InitLabeling(unsigned char *input, int *labels, size_t width, size_t height)
{
  // Calculate the x and y coordinates of the pixel this thread is responsible for
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // Check if the coordinates are within the image bounds
  if (x >= width || y >= height)
    return;

  // Calculate the 1D index of the pixel
  int idx = y * width + x;

  // Initialize labels: 255 (foreground) gets its own label (its index), others get -1
  labels[idx] = input[idx] == 255 ? idx : -1;
}

/**
 * @brief CUDA kernel for label propagation in connected component labeling.
 *
 * @param labels Pointer to labels data.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void LabelPropagation(int *labels, size_t width, size_t height)
{
  // Calculate the x and y coordinates of the pixel this thread is responsible for
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // Check if the coordinates are within the image bounds
  if (x >= width || y >= height)
    return;

  // Calculate the 1D index of the pixel
  int idx = y * width + x;
  int &label = labels[idx];

  // If the label is -1 (background), do nothing
  if (label == -1)
    return;

  int new_label = label;

  // Define offsets for 8-connected neighbors
  int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
  int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

  // Propagate the minimum label among the neighbors
  for (int i = 0; i < 8; ++i)
  {
    int nx = x + dx[i];
    int ny = y + dy[i];
    if (nx >= 0 && ny >= 0 && nx < width && ny < height)
    {
      int nidx = ny * width + nx;
      if (labels[nidx] != -1)
        new_label = min(new_label, labels[nidx]);
    }
  }

  // Update the label of the current pixel
  label = new_label;
}

/**
 * @brief CUDA kernel to flatten labels in connected component labeling.
 *
 * @param labels Pointer to labels data.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void FlattenLabels(int *labels, size_t width, size_t height)
{
  // Calculate the x and y coordinates of the pixel this thread is responsible for
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // Check if the coordinates are within the image bounds
  if (x >= width || y >= height)
    return;

  // Calculate the 1D index of the pixel
  int idx = y * width + x;
  int label = labels[idx];

  // If the label is -1 (background), do nothing
  if (label == -1)
    return;

  // Flatten the label by propagating the minimum label value
  while (label != labels[label])
    label = labels[label];

  // Update the label of the current pixel
  labels[idx] = label;
}

/**
 * @brief CUDA kernel to calculate segment lengths of a path.
 *
 * @param path Pointer to input path points.
 * @param lengths Pointer to output lengths array.
 * @param path_size Size of the path.
 */
__global__ void CalculateSegmentLengths(const IntPoint_t *path, double *lengths, size_t path_size)
{
  // Calculate the index of the current point this thread is responsible for
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // If the current point is within the path (excluding the last point)
  if (idx < path_size - 1)
  {
    // Calculate the length between consecutive points
    lengths[idx] = DeviceHypot(path[idx + 1].x - path[idx].x, path[idx + 1].y - path[idx].y);
  }
  else if (idx == path_size - 1)
  {
    // Last segment has no length
    lengths[idx] = 0.0;
  }
}

/**
 * @brief CUDA kernel to identify boundary pixels in a labeled image.
 *
 * @param labels Pointer to input labels data.
 * @param boundary_pixels Pointer to output boundary pixels array.
 * @param boundary_count Pointer to the number of boundary pixels.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void IdentifyBoundaryPixels(const int *labels, BoundaryPixel *boundary_pixels, int *boundary_count, size_t width, size_t height)
{
  // Calculate the x and y coordinates of the pixel this thread is responsible for
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the coordinates are within the image bounds
  if (x >= width || y >= height)
    return;

  // Calculate the 1D index of the pixel
  int idx = y * width + x;
  int label = labels[idx];

  // If the label is -1 (background), do nothing
  if (label == -1)
    return;

  // Check if this pixel is a boundary pixel
  bool is_boundary = false;
  // Define offsets for 4-connected neighbors
  int dx[] = {-1, 0, 1, 0};
  int dy[] = {0, -1, 0, 1};

  for (int i = 0; i < 4; ++i)
  {
    int nx = x + dx[i];
    int ny = y + dy[i];

    // Check if the neighbor is out of bounds or has a different label
    if (nx < 0 || ny < 0 || nx >= width || ny >= height || labels[ny * width + nx] != label)
    {
      is_boundary = true;
      break;
    }
  }

  if (is_boundary)
  {
    // Atomic addition to increment the boundary count
    int count = atomicAdd(boundary_count, 1);
    // Add the boundary pixel to the list
    boundary_pixels[count] = {label, {x, y}};
  }
}

/**
 * @brief Label connected components in a binary image using CUDA.
 *
 * @param input The input binary image (cv::Mat1b).
 * @return std::vector<int> The labeled image as a flat vector.
 */
std::vector<int> LabelComponents(const cv::Mat1b &input)
{
  int *cuda_labels;
  unsigned char *cuda_output;

  auto size = input.size();
  size_t n_pixels = size.width * size.height;
  size_t size_labels = n_pixels * sizeof(int);
  size_t size_output = n_pixels * sizeof(unsigned char);

  // Allocate memory on the device
  cudaMalloc(&cuda_labels, size_labels);
  cudaMalloc(&cuda_output, size_output);

  // Copy the input image data to the device
  cudaMemcpy(cuda_output, input.data, size_output, cudaMemcpyHostToDevice);

  // Define the grid and block dimensions
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((size.width + threads_per_block.x - 1) / threads_per_block.x, (size.height + threads_per_block.y - 1) / threads_per_block.y);

  // Initialize labels
  InitLabeling<<<num_blocks, threads_per_block>>>(cuda_output, cuda_labels, size.width, size.height);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
  }

  // Iterative label propagation and flattening
  for (int i = 0; i < 100; ++i)
  {
    LabelPropagation<<<num_blocks, threads_per_block>>>(cuda_labels, size.width, size.height);
    cudaDeviceSynchronize();
    FlattenLabels<<<num_blocks, threads_per_block>>>(cuda_labels, size.width, size.height);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
  }

  // Copy the labeled image back to the host
  std::vector<int> output(n_pixels);
  cudaMemcpy(output.data(), cuda_labels, size_labels, cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(cuda_labels);
  cudaFree(cuda_output);

  return output;
}

/**
 * @brief Extract contours from a labeled image using CUDA.
 *
 * @param labels The labeled image as a flat vector.
 * @param width Width of the image.
 * @param height Height of the image.
 * @return std::vector<std::vector<IntPoint_t>> The extracted contours.
 */
std::vector<std::vector<IntPoint_t>> ExtractContours(const std::vector<int> &labels, size_t width, size_t height)
{
  // Allocate memory on the GPU
  int *d_labels;
  BoundaryPixel *d_boundary_pixels;
  int *d_boundary_count;
  size_t labels_size = width * height * sizeof(int);
  size_t max_boundary_pixels = width * height; // Maximum possible number of boundary pixels
  size_t boundary_pixels_size = max_boundary_pixels * sizeof(BoundaryPixel);

  cudaMalloc(&d_labels, labels_size);
  cudaMalloc(&d_boundary_pixels, boundary_pixels_size);
  cudaMalloc(&d_boundary_count, sizeof(int));

  // Copy the labels to the device and initialize boundary count to zero
  cudaMemcpy(d_labels, labels.data(), labels_size, cudaMemcpyHostToDevice);
  cudaMemset(d_boundary_count, 0, sizeof(int));

  // Define the grid and block dimensions
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

  // Launch the kernel to identify boundary pixels
  IdentifyBoundaryPixels<<<num_blocks, threads_per_block>>>(d_labels, d_boundary_pixels, d_boundary_count, width, height);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
  }

  // Copy the boundary count and boundary pixels back to the host
  int boundary_count;
  cudaMemcpy(&boundary_count, d_boundary_count, sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<BoundaryPixel> boundary_pixels(boundary_count);
  cudaMemcpy(boundary_pixels.data(), d_boundary_pixels, boundary_count * sizeof(BoundaryPixel), cudaMemcpyDeviceToHost);

  // Free the allocated memory on the GPU
  cudaFree(d_labels);
  cudaFree(d_boundary_pixels);
  cudaFree(d_boundary_count);

  // Group boundary pixels by labels and create contours
  std::unordered_map<int, std::vector<IntPoint_t>> label_to_points;
  for (const auto &bp : boundary_pixels)
  {
    label_to_points[bp.label].emplace_back(bp.point);
  }

  // Convert the map to a vector of contours
  std::vector<std::vector<IntPoint_t>> contours;
  for (const auto &pair : label_to_points)
  {
    contours.push_back(pair.second);
  }

  return contours;
}