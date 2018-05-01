cmake_minimum_required(VERSION 3.5)
# This must be done before any language is set (ie before any project() or enable_language() command).
#set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
#set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(Parallel_kNN_with_SIFT)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR})

# header file search path
include_directories("/usr/local/include/" )
include_directories("/usr/include")
#include_directories("/usr/local/Cellar/glib/2.54.3/include/glib-2.0/")
#include_directories("/usr/local/Cellar/glib/2.54.3/lib/glib-2.0/include/")

link_directories("/usr/local/lib")
link_directories("/usr/lib")

add_executable(
        main
        main.cpp
        kNN.hpp
        kNN.cpp
        Dataset.hpp
        Dataset.cpp
)
target_link_libraries(main PUBLIC omp)
target_compile_options(main PUBLIC "-fopenmp")
