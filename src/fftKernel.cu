#include "cuda_kernel.h"

#include <cufft.h>
 

// Apply FFT to the path using cuFFT to determine Fourier coefficients
void apply_fft(const std::vector<Point>& path)
{
    size_t N = path.size();

    // Allocate device memory for the input and output data
    cufftComplex* d_in;
    cufftComplex* d_out;
    cudaMalloc(&d_in, sizeof(cufftComplex) * N);
    cudaMalloc(&d_out, sizeof(cufftComplex) * N);

    // Copy the path data to the device
    std::vector<cufftComplex> h_in(N);
    for (size_t i = 0; i < N; ++i)
    {
        h_in[i].x = static_cast<float>(path[i].x); // Real part
        h_in[i].y = static_cast<float>(path[i].y); // Imaginary part
    }
    cudaMemcpy(d_in, h_in.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    // Create a cuFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    // Execute the FFT
    cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD);

    // Copy the results back to the host
    std::vector<cufftComplex> h_out(N);
    cudaMemcpy(h_out.data(), d_out, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);

    // Output the Fourier coefficients
    std::cout << "Fourier Coefficients: " << std::endl;
    for (size_t i = 0; i < N; ++i)
    {
        std::cout << "Coefficient " << i << ": (" << h_out[i].x << ", " << h_out[i].y << ")" << std::endl;
    }

    // Destroy the cuFFT plan and free device memory
    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
}