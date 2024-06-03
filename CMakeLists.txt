cmake_minimum_required(VERSION 3.10)
project(canny_cluster LANGUAGES CUDA CXX)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(INC "${CMAKE_CURRENT_SOURCE_DIR}/inc")
set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(CUDA_SAMPLES_INC "${LIB}/cuda-samples/Common")
set(ARGPARSE_INC "${LIB}/argparse/include")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe --display_error_number --diag-suppress 611")

add_library(cuda_kernel ${SRC}/imageKernel.cu ${SRC}/clusteringKernel.cu ${SRC}/extra.cu)
target_include_directories(cuda_kernel PUBLIC ${CUDA_SAMPLES_INC} ${INC} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set_property(TARGET cuda_kernel PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

add_executable(canny_cluster ${SRC}/main.cpp)

target_include_directories(canny_cluster PUBLIC ${INC} ${ARGPARSE_INC} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(canny_cluster ${OpenCV_LIBS} cuda_kernel)

