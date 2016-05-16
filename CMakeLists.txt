cmake_minimum_required(VERSION 3.2)
project(GeneticCirclism)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(gc src/main.cc)
target_link_libraries(gc ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
