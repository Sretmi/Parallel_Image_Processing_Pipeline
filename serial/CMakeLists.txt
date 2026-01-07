add_executable(serial_app main.cpp)
target_link_libraries(serial_app PRIVATE ${OpenCV_LIBS})
target_include_directories(serial_app PRIVATE ${OpenCV_INCLUDE_DIRS})