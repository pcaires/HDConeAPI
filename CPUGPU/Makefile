CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17  -DMKL_ILP64  -I"${MKLROOT}/include"
LDFLAGS =  -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

all:	main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o hdc main.o

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

run:
	./hdc

clean:
	rm -f hdc; rm -f *.o
