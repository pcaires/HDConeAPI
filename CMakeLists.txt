cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(HDConeAPI)

find_package(IntelSYCL CONFIG REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

add_link_options(-lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lOpenCL -lpthread -lm -ldl)

add_definitions(-DPROJECT_PATH_CMAKE=\"${PROJECT_SOURCE_DIR}\")

add_subdirectory(CPUGPU)
