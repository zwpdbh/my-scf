cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")

project(xiaoxin_knn)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR})


# header files folder
include_directories("/usr/local/include/" "/usr/include")

# dynamicl libraries folder
link_directories("/usr/local/lib" "/usr/lib")
link_directories("/usr/local/lib/gcc/7")


# ./bin/scf_main 5 16 32 0 0 features/_DSC0%s.jpg features/gps-sift-small.csv
add_executable(
        scf_main
        main.cpp
        main.h
        common.h
        csv_parser.cpp
        csv_parser.h
        csv_parser.cpp
        sdc_index.cpp
        sdc_index.h
        sdc_kmeans.h
)
target_link_libraries(scf_main PUBLIC flann omp)
target_compile_options(scf_main PUBLIC "-fopenmp")
