add_executable(openmp_app main.cpp)
target_link_libraries(openmp_app PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_compile_options(openmp_app PRIVATE ${OpenMP_CXX_FLAGS})