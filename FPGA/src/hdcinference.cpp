#include <CL/sycl.hpp>
#include <chrono>
#include <ctime>
#include <random>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <math.h>  
#include "pipe_array.hpp"
#include "unrolled_loop.hpp"
//#include <sycl/ext/intel/fpga_extensions.hpp>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#define chunk 16

#define numPipes 25

//#define D 2000
#define NCLASSMAX 10
#define NFEATMAX 784
#define DIVNFEATMAX 49 // 16 * 50 = 800 
#define DMAX 2000
#define DIVDMAX 125 // 16 * 128 = 2048
#if FPGA_EMULATOR
#define NTRAIN 6000
#else
#define NTRAIN 60000
#endif
#define NTEST 5575
#define INTERIM 5
#define PERC_UPDATE 0.2
#define NUMREGEN 200
#define MAXBATCH 256

#define K DMAX / numPipes

#define MIN(a,b) a<b?a:b

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

#define PRINTF(format, ...)                                   \
  {                                                           \
    static const CL_CONSTANT char _format[] = format;         \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__);\
  }

typedef std::chrono::high_resolution_clock::time_point timePoint;

using namespace sycl;
template<int i> class encodeKernel;
class classifyKernel1;
class classifykernel2;
class producer;
class consumer;
template<int i> class bundler;
using namespace std::chrono;

typedef struct {
	std::vector<std::vector<float>> data;
	int* labels;
	int32_t numFeatures;
	int32_t numClasses;
} Data;


double elapsedTime(timePoint start, timePoint end) {
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
  return time_span.count();
}

std::vector<float> normalize(std::vector<float> input) {
	double inner = std::inner_product(input.begin(), input.end(), input.begin(), 0.0);
	double norm = std::sqrt(inner);
	for(int i = 0; i < input.size(); i++) {
		input[i] = input[i] / norm;
	}

	return input;

}
void normalizeClasses(float *classes, int numClasses, int numDim) {
    for(int i = 0; i < numClasses; i++) {
        float norm = 0.0;
        for(int j = 0; j < numDim; j++) {
            norm += classes[i * numDim + j] * classes[i * numDim + j];
        }
        norm = std::sqrt(norm);
        for(int j = 0; j < numDim; j++) {
            classes[i * numDim + j] /= norm;
        }
    }
}
int infer(float *bases, float *classes, std::vector<std::vector<float>> data, int *labels, int numFeats, int numDim, int numClasses) {
    int correct = 0;
    for(int i = 0; i < data.size(); i++) {
        float encoded[numDim];
        for(int j = 0; j < numDim; j++) {
            float sum = 0.0;
            for(int k = 0; k < numFeats; k++) {
                sum += bases[j * numFeats + k] * data[i][k];
            }
            encoded[j] = cosf(sum);
        }
        float max = 0.0;
        float maxInd = 0;
        for(int j = 0; j < numClasses; j++) {
            float sum = 0;
            for(int k = 0; k < numDim; k++) {
                sum += classes[j * numDim + k] * encoded[k];
            }
            if(sum >= max) {
                max = sum;
                maxInd = j;
            }
        }
        if(maxInd == labels[i]) {
            correct++;
        }
    }
    return correct;
}

void validateEncoding(float *bases, std::vector<std::vector<float>> data, float *encoded, int numDim, int numFeats, int start, int numData) {
    float epsilon = 0.00005;
     for(int i = 0; i < numData; i++) {
        float enc[numDim];
        for(int j = 0; j < numDim; j++) {
            float sum = 0.0;
            for(int k = 0; k < numFeats; k++) {
                sum += bases[j * numFeats + k] * data[i + start][k];
            }
            enc[j] = cosf(sum);
        }
        for(int j = 0; j < numDim; j++) {
            if(fabs(enc[j] - encoded[(i) * numDim + j]) > epsilon) {
                std::cout << "Error in encoding for data " << i << "\n";
                break;
            }
        }
    }
}

void validateBundling(float *classes, float *referenceClasses, float *bases, std::vector<std::vector<float>> data, int *labels, int numDim, int numData, int numClasses, int numFeats) {
    float epsilon = 0.00005;
    
    std::vector<std::vector<float>> encoded(numData);
    for(int i = 0; i < numData; i++) {
        std::vector<float> temp(numDim);
        for(int j = 0; j < numDim; j++) {
            float sum = 0.0;
            for(int k = 0; k < numFeats; k++) {
                sum += bases[j * numFeats + k] * data[i][k];
            }
            temp[j] = cosf(sum);
        }
        encoded[i] = temp;
    }
    for(int i = 0; i < numDim * numClasses; i++) {
        referenceClasses[i] = 0.0;
    }
    for(int i = 0; i < numData; i++) {
        int index = labels[i];
        for(int j = 0; j < numDim; j++) {
            referenceClasses[index * numDim + j] += encoded[i][j];
        }
    }
    normalizeClasses(referenceClasses, numClasses, numDim);
    for(int i = 0; i < numClasses; i++) {
        for(int j = 0; j < numDim; j++) {
            if(fabs(classes[i * numDim + j] - referenceClasses[i * numDim + j]) > epsilon) {
                std::cout << "Error in class " << i << "\n";
                std::cout << "\tClass " << i << " Dim " << j << "\t" << fabs(classes[i * numDim + j] - referenceClasses[i * numDim + j]) << "\n";
                break;
            }
        }
    }
}


void validateInference(float *bases, float *classes, std::vector<std::vector<float>> data, int *referencePreds, int numData, int numDim, int numFeats, int numClasses) {
    for(int i = 0; i < numData; i++) {
        float encoded[numDim];
        for(int j = 0; j < numDim; j++) {
            float sum = 0.0;
            for(int k = 0; k < numFeats; k++) {
                sum += bases[j * numFeats + k] * data[i][k];
            }
            encoded[j] = cosf(sum);
        }
        float max = 0.0;
        float maxInd = 0;
        for(int j = 0; j < numClasses; j++) {
            float sum = 0;
            for(int k = 0; k < numDim; k++) {
                sum += classes[j * numDim + k] * encoded[k];
            }
            if(sum >= max) {
                max = sum;
                maxInd = j;
            }
        }
        if(maxInd != referencePreds[i]) {
            std::cout << "incorrect:\t" << i << "\t" <<  maxInd << "\t" << referencePreds[i]<< std::endl;
        }
    }
}

Data readData(char *filename, queue &q) {
	std::ifstream testFile(filename, std::ifstream::binary);

	char *holder = (char *)malloc(4 * sizeof(char));
	testFile.read(holder, 4 * sizeof(char));
	int32_t numFeatures;
	memcpy(&numFeatures, &holder[0], sizeof(numFeatures));
	testFile.read(holder, 4 * sizeof(char));
	int32_t numClasses;
	memcpy(&numClasses, &holder[0], sizeof(numClasses));
	std::vector<std::vector<float>> testData;
	std::vector<int> testLabels;
	while(testFile.good()) {
		std::vector<float> vect(numFeatures, 0.0);
		bool zero = true;
		for(int i = 0; i < numFeatures; i++) {
			testFile.read(holder, 4 * sizeof(char));
			float val;
			memcpy(&val, &holder[0], sizeof(val));
			vect[i] = val;
			if(val != 0.0) {
				zero = false;
			}
		} 
		testFile.read(holder, 4 * sizeof(char));
		int label;
		memcpy(&label, &holder[0], sizeof(label));
		if(!zero) {
			vect = normalize(vect);
		testData.push_back(vect);
		testLabels.push_back(label);
		}
	}
	free(holder);
	testFile.close();
    int numData = ((testLabels.size() / MAXBATCH) + 1) * MAXBATCH;
    int *labels = malloc_device<int>(numData, q);
    q.memcpy(labels, testLabels.data(), sizeof(int) * testLabels.size()).wait();

	Data ret;
	ret.data = testData;
	ret.labels = labels;
	ret.numClasses = numClasses;
	ret.numFeatures = numFeatures;
	return ret;
}

class encodePipe;
class bundlePipe;
class maxPipe;
using toEncode = PipeArray<encodePipe, float, 4 * NFEATMAX, numPipes>;
using toClassify = PipeArray<bundlePipe, float, 8 * K, numPipes>;
using toConsume = PipeArray<maxPipe, float, 8 * K, numPipes>;
template<int npipes>
void produce(queue &q, float* input, int numData, int numFeats) {
    const int numBatches = NTRAIN / MAXBATCH + 1;
    q.submit([&](auto &h) { 
        h.template single_task<producer>([=]() [[intel::kernel_args_restrict]] {
            float localInput[MAXBATCH][NFEATMAX];
            
            for(int j = 0, k = 0, l = 0; j < MAXBATCH * NFEATMAX; j++) {
                localInput[k][l] = input[j];
                l++;
                if(l == NFEATMAX) {
                    l = 0;
                    k++;
                }
            }
            for(int j = 0; j < MAXBATCH; j++) {
                for(int k = 0; k < NFEATMAX; k++) {
                    
                    impu::UnrolledLoop<npipes>([&] (auto l) {
                        toEncode::PipeAt<l>::write(localInput[j][k]);
                    });
                }
            }
        });
    });
}

template<int pipeId>
void encode(queue &q, float *bases) {
    q.submit([&](auto &h) {
        h.template single_task<encodeKernel<pipeId>>([=]() [[intel::kernel_args_restrict]] {
            float localBases[K][NFEATMAX];
            
            for(int ij = 0, i = 0, j = 0; ij < K * NFEATMAX; ij++) {
                localBases[i][j] = bases[pipeId * K * NFEATMAX + ij];
                j++;
                if(j == NFEATMAX) {
                    i++;
                    j = 0;
                }
            }
            for(int i = 0; i < MAXBATCH; i++) {
                float data[NFEATMAX];
                for(int j = 0; j < NFEATMAX; j++) {
                    data[j] = toEncode::PipeAt<pipeId>::read();
                }
                float output[K];
                for(int j = 0; j < K; j++) {
                    float sum = 0.0f;
                    #pragma unroll 16
                    for(int k = 0; k < NFEATMAX; k++) {
                        sum += localBases[j][k] * data[k];
                    }
                    output[j] = cosf(sum);
                }
                for(int j = 0; j < K; j++) {
                    toClassify::PipeAt<pipeId>::write(output[j]);
                }
            }
        });
    });
}

template<int pipeId> 
void bundle(queue &q, float *classes) {
    q.submit([&](auto &h) {
        h.template single_task<bundler<pipeId>>([=]() [[intel::kernel_args_restrict]] {
            float localClasses[NCLASSMAX][K];
            
            for(int i = 0; i < NCLASSMAX; i++) {
                for(int j = 0; j < K; j++) {
                    localClasses[i][j] = classes[i * DMAX + pipeId * K + j];
                }
            }
            for(int i = 0; i < MAXBATCH; i++) {
                float encoded[K];
                for(int j = 0; j < K; j++) {
                    encoded[j] = toClassify::PipeAt<pipeId>::read();
                }
                float max[NCLASSMAX];
                for(int k = 0; k < NCLASSMAX; k++) {
                    float sum = 0.0f;
                    for(int j = 0; j < K; j++) {
                        sum += localClasses[k][j] * encoded[j];
                    }
                    max[k] = sum;
                }
                for(int j = 0; j < NCLASSMAX; j++) {
                    toConsume::PipeAt<pipeId>::write(max[j]);
                }
            }

        });
    });
}

template<int npipes>
void consume(queue &q, int *output, int numData) {
    const int numBatches = NTRAIN / MAXBATCH + 1;
    q.submit([&](auto &h) { 
        h.template single_task<consumer>([=]() [[intel::kernel_args_restrict]] {
            int localOutput[MAXBATCH];
            
            for(int j = 0; j < MAXBATCH; j++) {
                    float maxes[npipes][NCLASSMAX];
                    impu::UnrolledLoop<npipes>([&maxes] (auto l) {
                        for(int i = 0; i < NCLASSMAX; i++) {
                            maxes[l][i] = toConsume::PipeAt<l>::read();
                        }
                    });
                    float max[NCLASSMAX];
                    for(int i = 0; i < NCLASSMAX; i++) {
                        float sum = 0.0f;
                        for(int k = 0; k < npipes; k++) {
                            sum += maxes[k][i];
                        }
                        max[i] = sum;
                    }
                    float totalMax = 0.0;
                    int maxInd = 0;
                    for(int i = 0; i < NCLASSMAX; i++) {
                        if(max[i] > totalMax) {
                            totalMax = max[i];
                            maxInd = i;
                        }
                    }
                    localOutput[j] = maxInd;
            }
            for(int i = 0; i < MAXBATCH; i++) {
                output[i] = localOutput[i];
            }
        });
    });
}
#if FPGA_EMULATOR
	ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
	ext::intel::fpga_selector d_selector;
#else
	default_selector d_selector;
#endif

void initialize(float *bases, float *classes, float *data, int numFeats, int numClass, int numDim, int numData) {

    std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<> d{0.0, 1.0};
    for(int i = 0; i < numFeats * numDim; i++) {
        bases[i] = d(gen);
    }
    for(int i = 0; i < numClass * numDim; i++) {
        classes[i] = d(gen);
    }
    for(int i = 0; i < numData * numFeats; i++) {
        data[i] = d(gen);
    }
}

class encodeID;
class classifyID;

int main(int argc, char *argv[]) {
    int numBatches = 100;
    for(int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        
        if(arg == "--help" || arg == "-h") {
            std::cout << "USAGE: ./hdc.fpga [--batches=<int>]\n";
            return 0;
        }
        
        std::string after = arg.substr(arg.find("=") + 1);
        if(arg.find("--batches=") == 0) {
            numBatches = atoi(after.c_str());
        }

    }
    queue q(d_selector);
    
    int numFeats, numClasses, numDim, batch;
    
    numFeats = 784;
    numClasses = 10;
    numDim = 2000;
    batch = MAXBATCH;
    char* trainFile = strdup("mnist_train.choir_dat");
	Data train = readData(trainFile, q);
    int numData = train.data.size();
    float *bases = static_cast<float *>(malloc(sizeof(float) * numFeats * numDim));
    float *classes = static_cast<float *>(malloc(sizeof(float) * numClasses * numDim));
    float *localClasses = static_cast<float *>(malloc(sizeof(float) * numClasses * numDim));
    float *data = static_cast<float *>(malloc(sizeof(float) * numData * numFeats));
    float *tempEncoded = static_cast<float *>(malloc(sizeof(float) * numDim * batch));
    initialize(bases, classes, data, numFeats, numClasses, numDim, batch * numBatches);
    for(int i = 0; i < numData; i++) {
        for(int j = 0; j < numFeats; j++) {
            data[i * numFeats + j] = train.data[i][j];
        }
    }

    numBatches = (numData / MAXBATCH) + 1;
    float *deviceBases = malloc_device<float>(numFeats * numDim, q);
    float *deviceClasses = malloc_device<float>(numClasses * numDim, q);
    float *deviceData = malloc_device<float>(numBatches * batch * numFeats, q);
    float *intermeds1 = malloc_device<float>(batch * numDim, q);
    float *intermeds2 = malloc_device<float>(batch * numDim, q);
    int *preds = malloc_device<int>(batch, q);
    int *deviceLabels = train.labels;
    
    auto copyStart = high_resolution_clock::now();
    q.memcpy(deviceBases, bases, sizeof(float) * numFeats * numDim).wait();
    q.memcpy(deviceClasses, classes, sizeof(float) * numClasses * numDim).wait();
    q.memcpy(deviceData, data, sizeof(float) * numData * numFeats).wait();
    auto copyEnd = high_resolution_clock::now();
    int cur = 0;
    impu::UnrolledLoop<numPipes>([&](auto index) {
        encode<index>(q, deviceBases);
        bundle<index>(q, deviceClasses);
    });
    auto start = high_resolution_clock::now();
    produce<numPipes>(q, deviceData, numData, numFeats);
    consume<numPipes>(q, preds, numData);
    q.wait();
    
    auto end = high_resolution_clock::now();
    
    printf("TIME: %f\n", elapsedTime(start, end));
    
    int *hostPreds = static_cast<int *>(malloc(sizeof(int) * batch));
    q.memcpy(hostPreds, preds, sizeof(int) * batch).wait();
    validateInference(bases, classes, train.data, hostPreds, batch, numDim, numFeats, numClasses);
        
    sycl::free(deviceBases, q);
    sycl::free(deviceClasses, q);
    sycl::free(deviceData, q);
    sycl::free(intermeds1, q);
    sycl::free(intermeds2, q);
    sycl::free(deviceLabels, q);
    sycl::free(preds, q);
    
    free(bases);
    free(classes);
    free(data);
    free(tempEncoded);
    free(localClasses);
    free(hostPreds);

    return 0;
}