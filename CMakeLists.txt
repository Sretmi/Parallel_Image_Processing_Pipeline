cmake_minimum_required(VERSION 3.10)

project(ImageProcessingPipeline)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(serial)
add_subdirectory(openmp)
add_subdirectory(mpi)