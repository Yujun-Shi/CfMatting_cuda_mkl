cmake_minimum_required(VERSION 2.8)
project(CfMatting CXX)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


set(INC_DIRS $ENV{CUDA_HOME}/include include)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O3")

include_directories(${INC_DIRS})

set(OPENMP_FLAG "-fopenmp -std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  -gencode arch=compute_61,code=sm_61
                                        -gencode arch=compute_61,code=compute_61)

cuda_add_executable(CfMatting src/CfMatting.cu src/utils.cu src/main.cpp)

target_link_libraries(CfMatting ${OpenCV_LIBS} ${CUDA_LIBRARY} ${CUDA_cusparse_LIBRARY}
                    mkl_intel_lp64 mkl_core mkl_gnu_thread iomp5 pthread m dl)

target_compile_options(CfMatting
    PUBLIC -I${MKLROOT}/include)

