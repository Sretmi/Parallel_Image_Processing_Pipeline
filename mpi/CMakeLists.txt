add_executable(mpi_app main.cpp)
target_link_libraries(mpi_app PRIVATE ${OpenCV_LIBS} MPI::MPI_CXX)
target_include_directories(mpi_app PRIVATE ${MPI_CXX_INCLUDE_PATH})