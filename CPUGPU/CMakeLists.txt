add_executable(CPUGPU main.cpp)
add_sycl_to_target(TARGET CPUGPU SOURCES main.cpp)
