rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_CXX_COMPILER=icpx
make CPUGPU && ./CPUGPU/CPUGPU