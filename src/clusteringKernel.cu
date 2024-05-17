#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_kernel.h"

// Define a struct to hold boundary pixels
struct BoundaryPixel
{
    int label;
    Point point;
};

// Device-compatible hypot function
__device__ double device_hypot(double x, double y)
{
    return sqrt(x * x + y * y);
}

__global__ void InitLabeling(unsigned char *input, int *labels, size_t width, size_t height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    labels[idx] = input[idx] == 255 ? idx : -1;
}

__global__ void LabelPropagation(int *labels, size_t width, size_t height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int label = labels[idx];
    if (label == -1)
        return;

    int new_label = label;

    // 8-connected neighbors
    int dx[] = {-1,  0,  1, -1, 1, -1, 0, 1};
    int dy[] = {-1, -1, -1,  0, 0,  1, 1, 1};

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

    labels[idx] = new_label;
}

__global__ void FlattenLabels(int *labels, size_t width, size_t height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int label = labels[idx];
    if (label == -1)
        return;

    // Flatten the label by propagating the minimum label value
    while (label != labels[label])
        label = labels[label];

    labels[idx] = label;
}

__global__ void RemoveSmallClusters(int *labels, unsigned char *output, size_t width, size_t height, int min_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int label = labels[idx];

    if (label == -1)
    {
        output[idx] = 0;
        return;
    }

    // Count the number of pixels in each label
    __shared__ int cluster_size[1024];

    if (threadIdx.x == 0 && threadIdx.y == 0)
        memset(cluster_size, 0, sizeof(cluster_size));
    __syncthreads();

    atomicAdd(&cluster_size[label % 1024], 1);
    __syncthreads();

    int size = cluster_size[label % 1024];

    if (size < min_size)
        output[idx] = 0;
    else
        output[idx] = 255;
}

__global__ void calculateSegmentLengthsKernel(const Point* path, double* lengths, size_t path_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < path_size - 1)
    {
        lengths[i] = device_hypot(path[i + 1].x - path[i].x, path[i + 1].y - path[i].y);
    }
    else if (i == path_size - 1)
    {
        lengths[i] = 0.0; // Last segment has no length
    }
}


__global__ void interpolatePath(const Point* path, const double* lengths, Point* interpolated_path, size_t path_size, size_t num_points, double step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_points)
        return;

    if (i == 0)
    {
        interpolated_path[i] = path[0];
        return;
    }

    if (i == num_points - 1)
    {
        interpolated_path[i] = path[path_size - 1];
        return;
    }

    double current_length = i * step;
    size_t current_index = 0;

    while (current_index < path_size - 1 && current_length > lengths[current_index + 1])
    {
        ++current_index;
    }

    double t = (current_length - lengths[current_index]) / (lengths[current_index + 1] - lengths[current_index]);
    interpolated_path[i].x = static_cast<int>((1 - t) * path[current_index].x + t * path[current_index + 1].x);
    interpolated_path[i].y = static_cast<int>((1 - t) * path[current_index].y + t * path[current_index + 1].y);
}

__global__ void parallelReduction(double* lengths, double* total_length, size_t path_size)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < path_size)
        sdata[tid] = lengths[i];
    else
        sdata[tid] = 0.0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(total_length, sdata[0]);
    }
}

// CUDA Kernel to identify boundary pixels
__global__ void identifyBoundaryPixels(const int *labels, BoundaryPixel *boundary_pixels, int *boundary_count, size_t width, size_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int label = labels[idx];

    if (label == -1)
        return;

    // Check if this pixel is a boundary pixel
    bool is_boundary = false;
    int dx[] = {-1, 0, 1, 0};
    int dy[] = {0, -1, 0, 1};

    for (int i = 0; i < 4; ++i)
    {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (nx < 0 || ny < 0 || nx >= width || ny >= height || labels[ny * width + nx] != label)
        {
            is_boundary = true;
            break;
        }
    }

    if (is_boundary)
    {
        int count = atomicAdd(boundary_count, 1);
        boundary_pixels[count] = {label, {x, y}};
    }
}

void label_components(unsigned char *input, int *output, size_t width, size_t height)
{
    int *cuda_labels;
    unsigned char *cuda_output;

    size_t n_pixels = width * height;
    size_t size_labels = n_pixels * sizeof(int);
    size_t size_output = n_pixels * sizeof(unsigned char);

    cudaMalloc(&cuda_labels, size_labels);
    cudaMalloc(&cuda_output, size_output);

    cudaMemcpy(cuda_output, input, size_output, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    InitLabeling<<<numBlocks, threadsPerBlock>>>(cuda_output, cuda_labels, width, height);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    for (int i = 0; i < 100; ++i) // Iterative label propagation
    {
        LabelPropagation<<<numBlocks, threadsPerBlock>>>(cuda_labels, width, height);
        cudaDeviceSynchronize();
        FlattenLabels<<<numBlocks, threadsPerBlock>>>(cuda_labels, width, height);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
        }
        
    }

    cudaMemcpy(output, cuda_labels, size_labels, cudaMemcpyDeviceToHost);

    cudaFree(cuda_labels);
    cudaFree(cuda_output);
}

// Nearest Neighbor TSP Solver
std::vector<Point> solve_tsp(const std::vector<Point>& points)
{
    std::vector<Point> path;
    if (points.empty()) return path;

    std::vector<bool> visited(points.size(), false);
    path.push_back(points[0]);
    visited[0] = true;

    while (path.size() < points.size())
    {
        const Point& last = path.back();
        int nearest_idx = -1;
        double nearest_dist = std::numeric_limits<double>::max();

        for (size_t i = 0; i < points.size(); ++i)
        {
            if (!visited[i])
            {
                double dist = std::hypot(points[i].x - last.x, points[i].y - last.y);
                if (dist < nearest_dist)
                {
                    nearest_dist = dist;
                    nearest_idx = i;
                }
            }
        }

        path.push_back(points[nearest_idx]);
        visited[nearest_idx] = true;
    }

    return path;
}

// Function to extract contours using CUDA
std::vector<std::vector<Point>> extract_contours(const int *labels, size_t width, size_t height)
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

    cudaMemcpy(d_labels, labels, labels_size, cudaMemcpyHostToDevice);
    cudaMemset(d_boundary_count, 0, sizeof(int));

    // Launch the kernel to identify boundary pixels
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    identifyBoundaryPixels<<<numBlocks, threadsPerBlock>>>(d_labels, d_boundary_pixels, d_boundary_count, width, height);
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
    std::unordered_map<int, std::vector<Point>> label_to_points;
    for (const auto& bp : boundary_pixels)
    {
        label_to_points[bp.label].emplace_back(bp.point);
    }

    std::vector<std::vector<Point>> contours;
    for (const auto& pair : label_to_points)
    {
        contours.push_back(pair.second);
    }

    return contours;
}

void remove_small_clusters(unsigned char *input, unsigned char *output, size_t width, size_t height, int min_size)
{
    int *cuda_labels;
    unsigned char *cuda_output;

    size_t n_pixels = width * height;
    size_t size_labels = n_pixels * sizeof(int);
    size_t size_output = n_pixels * sizeof(unsigned char);

    cudaMalloc(&cuda_labels, size_labels);
    cudaMalloc(&cuda_output, size_output);

    cudaMemcpy(cuda_output, input, size_output, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    InitLabeling<<<numBlocks, threadsPerBlock>>>(cuda_output, cuda_labels, width, height);
    cudaDeviceSynchronize();

    for (int i = 0; i < 100; ++i) // Iterative label propagation
    {
        LabelPropagation<<<numBlocks, threadsPerBlock>>>(cuda_labels, width, height);
        cudaDeviceSynchronize();
        FlattenLabels<<<numBlocks, threadsPerBlock>>>(cuda_labels, width, height);
        cudaDeviceSynchronize();
    }

    RemoveSmallClusters<<<numBlocks, threadsPerBlock>>>(cuda_labels, cuda_output, width, height, min_size);
    cudaDeviceSynchronize();

    cudaMemcpy(output, cuda_output, size_output, cudaMemcpyDeviceToHost);

    cudaFree(cuda_labels);
    cudaFree(cuda_output);
}

void interpolate_path(const std::vector<Point>& path, std::vector<Point>& interpolated_path, size_t num_points)
{
    size_t path_size = path.size();
    if (path_size <= 1 || num_points == 0)
        return;

    // Allocate memory on the GPU
    Point* d_path;
    Point* d_interpolated_path;
    double* d_lengths;
    double* d_total_length;

    size_t path_bytes = path_size * sizeof(Point);
    size_t lengths_bytes = path_size * sizeof(double);
    size_t interpolated_path_bytes = num_points * sizeof(Point);

    cudaMalloc(&d_path, path_bytes);
    cudaMalloc(&d_interpolated_path, interpolated_path_bytes);
    cudaMalloc(&d_lengths, lengths_bytes);
    cudaMalloc(&d_total_length, sizeof(double));

    cudaMemcpy(d_path, path.data(), path_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_total_length, 0, sizeof(double));

    // Launch the segment length calculation kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (path_size + threadsPerBlock - 1) / threadsPerBlock;
    calculateSegmentLengthsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_path, d_lengths, path_size);
    cudaDeviceSynchronize();
   
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    // Launch the parallel reduction kernel
    parallelReduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_lengths, d_total_length, path_size);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    // Copy the total length back to the host
    double total_length;
    cudaMemcpy(&total_length, d_total_length, sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate the cumulative lengths
    std::vector<double> cumulative_lengths(path_size);
    cudaMemcpy(cumulative_lengths.data(), d_lengths, lengths_bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 1; i < path_size; ++i)
    {
        cumulative_lengths[i] += cumulative_lengths[i - 1];
    }
    cudaMemcpy(d_lengths, cumulative_lengths.data(), lengths_bytes, cudaMemcpyHostToDevice);

    // Launch the interpolation kernel
    double step = total_length / (num_points - 1);
    blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    interpolatePath<<<blocksPerGrid, threadsPerBlock>>>(d_path, d_lengths, d_interpolated_path, path_size, num_points, step);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    // Copy the interpolated path back to the host
    interpolated_path.resize(num_points);
    cudaMemcpy(interpolated_path.data(), d_interpolated_path, interpolated_path_bytes, cudaMemcpyDeviceToHost);

    // Free the allocated memory on the GPU
    cudaFree(d_path);
    cudaFree(d_interpolated_path);
    cudaFree(d_lengths);
    cudaFree(d_total_length);
}