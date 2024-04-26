cmake_minimum_required(VERSION 3.10)

project(HDConeAPI)

find_package(IntelSYCL REQUIRED)

add_subdirectory(CPUGPU)
