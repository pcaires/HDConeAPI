add_executable(CPUGPU main.cpp)

target_link_libraries(CPUGPU PUBLIC MKL::MKL)

add_sycl_to_target(TARGET CPUGPU SOURCES main.cpp)
