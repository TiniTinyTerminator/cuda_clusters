cmake_minimum_required(VERSION 3.10)
project(fourier_series LANGUAGES CUDA CXX)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(INC "${CMAKE_CURRENT_SOURCE_DIR}/inc")
set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(CUDA_SAMPLES_INC "${LIB}/cuda-samples/Common")

# include_directories(fourier_series ${OpenCV_INCLUDE_DIRS})
# include_directories( ${INC})
# include_directories( ${CUDA_SAMPLES_INC})

add_library(cuda_kernel ${SRC}/imageKernel.cu ${SRC}/clusteringKernel.cu ${SRC}/fftKernel.cu)
target_include_directories(cuda_kernel PUBLIC ${CUDA_SAMPLES_INC} ${INC} ${CUDA_INCLUDE_DIRS})
set_property(TARGET cuda_kernel PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

add_executable(fourier_series ${SRC}/main.cpp)

target_include_directories(cuda_kernel PUBLIC ${OpenCV_INCLUDE_DIRS} ${INC})
target_link_libraries(cuda_kernel CUDA::cufft)

target_link_libraries(fourier_series ${OpenCV_LIBS} cuda_kernel)

