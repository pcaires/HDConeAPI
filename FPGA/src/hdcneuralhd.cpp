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
#include <malloc.h>
//#include <sycl/ext/intel/fpga_extensions.hpp>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#define chunk 16

//#define D 2000
#define NCLASSMAX 10
#define NFEATMAX 784
#define DIVNFEATMAX 49 // 16 * 50 = 800 
#define DMAX 2000
#define DIVDMAX 125 // 16 * 128 = 2048

#define NTRAIN 60000

#define NTEST 5575
#define INTERIM 5
#define PERC_UPDATE 0.2
#define NUMREGEN 200
#define MAXBATCH 256


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
class encodeKernel;
class classifyKernel;
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

std::vector<float>normalizeClasses(float *classes, int numClasses, int numDim) {
    std::vector<float> normed(numClasses * numDim);
    for(int i = 0; i < numClasses; i++) {
        float norm = 0.0;
        for(int j = 0; j < numDim; j++) {
            norm += classes[i * numDim + j] * classes[i * numDim + j];
        }
        norm = std::sqrt(norm);
        for(int j = 0; j < numDim; j++) {
            normed[i * numDim + j] = classes[i * numDim + j] / norm;
        }
    }
    return normed;
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

    int *labels = malloc_device<int>(testLabels.size(), q);
    q.memcpy(labels, testLabels.data(), sizeof(int) * testLabels.size()).wait();

	Data ret;
	ret.data = testData;
	ret.labels = labels;
	ret.numClasses = numClasses;
	ret.numFeatures = numFeatures;
	return ret;
}

event encode(queue &q, float *bases, float *inputData, float *out, int start, int D, int numFeat) {
    auto ret = q.submit([&] (handler &h) {
//         accessor input(inputData, h, read_only);
//         accessor base(bases, h, read_only);
//         accessor out(output, h, write_only, no_init);
        
        h.single_task<encodeKernel>([=]() [[intel::kernel_args_restrict]] {
            [[intel::numbanks(chunk), intel::bankwidth(4)]] float localBases[DMAX][NFEATMAX];
            [[intel::numbanks(chunk), intel::bankwidth(4)]] float localInput[MAXBATCH][NFEATMAX];
            float localOutput[MAXBATCH][DMAX];
            sycl::ext::intel::fpga_loop_fuse_independent([&] {
                
                for(int ij = 0, i = 0, j = 0; ij < DMAX * DIVNFEATMAX; ij++) {
                    #pragma unroll
                    for(int l = 0; l < chunk; l++) {
                        localBases[i][j * chunk + l] = bases[ij * chunk + l];
                    }
                    
                    j++;
                    if(j == DIVNFEATMAX) {
                        j = 0;
                        i++;
                    }
                }
                for(int ij = start * NFEATMAX, i = 0, j = 0; ij < (start + MAXBATCH) * NFEATMAX; ij++) {
                    localInput[i][j] = inputData[ij];
                    j++;
                    if(j == NFEATMAX) {
                        j = 0;
                        i++;
                    }
                }
            });
            
            for(int i = 0; i < MAXBATCH; i++) {
                #pragma unroll 8
                for(int j = 0; j < DMAX; j++) {
                    float sum = 0.0f;
                    #pragma unroll 16
                    for(int k = 0; k < NFEATMAX; k++) {
                        sum += (localBases[j][k] * localInput[i][k]);
                    }
                    localOutput[i][j] = cosf(sum);
                }
            }
            for(int ij = start * DMAX, i = 0, j = 0; ij < DMAX * (MAXBATCH + start); ij++) {
                out[ij] = localOutput[i][j];
                j++;
                if(j == DMAX) {
                    j = 0;
                    i++;
                }
            }
        });
    });
    return ret;
}


event classify(queue &q, float *classes, float *inputData, int *labels, int start, int numData, int D, int numClasses, int *correct){
    auto ret = q.submit([&](handler &h) {
//         accessor data(encodedData, h, read_only);
//         accessor klass(classes, h, read_only);
//         accessor out(output, h, write_only, no_init);
        
        h.single_task<classifyKernel>([=]() [[intel::kernel_args_restrict]]{
            [[intel::numbanks(chunk), intel::bankwidth(4)]] float localClasses[NCLASSMAX][DMAX];
            [[intel::numbanks(chunk), intel::bankwidth(4)]] float localInput[MAXBATCH][DMAX];
            [[intel::numbanks(chunk), intel::bankwidth(4)]] int localLabels[MAXBATCH];
            sycl::ext::intel::fpga_loop_fuse_independent([&] {
                for(int ij = start * DMAX, i = 0, j = 0; ij < (start + MAXBATCH) * DMAX; ij++) {
                    localInput[i][j] = inputData[ij];
                    j++;
                    if(j == DMAX) {
                        j = 0;
                        i++;
                    }
                }
                for(int ij = 0, i = 0, j = 0; ij < NCLASSMAX * DMAX; ij++) {
                    localClasses[i][j] = classes[ij];
                    j++;
                    if(j == DMAX) {
                        j = 0;
                        i++;
                    }
                }
                for(int i = 0; i < MAXBATCH; i++) {
                    localLabels[i] = labels[i + start];
                }
            });
            int corr = 0;
            for(int i = 0; i < MAXBATCH; i++) {
                float max[NCLASSMAX];
                if(i > numData) {
                    continue;
                }
                #pragma unroll 5
                for(int j = 0; j < NCLASSMAX; j++) {
                    float sum = 0.0f;
                    #pragma unroll 16
                    for(int k = 0; k < DMAX; k++) {
                        sum += localClasses[j][k] * localInput[i][k]; 
                    }
                    max[j] = sum;
                }
                int maxInd = -1;
                float realMax = -1.0;
                #pragma unroll
                for(int j = 0; j < NCLASSMAX; j++) {
                    if(max[j] > realMax) {
                        realMax = max[j];
                        maxInd = j;
                    }
                }
                int ans = localLabels[i];
                if(maxInd == ans) {
                    corr++;
                }
                float weight = 0.037;
                #pragma unroll chunk
                for(int j = 0; j < DMAX; j++) {
                    if(maxInd != ans) {
                        localClasses[ans][j] += weight * localInput[i][j];
                        localClasses[maxInd][j] -= weight * localInput[i][j];
                    }
                }
            }
            #pragma unroll chunk
            for(int ij = 0, i = 0, j = 0; ij < NCLASSMAX * DMAX; ij++) {
                classes[ij] = localClasses[i][j];
                j++;
                if(j == DMAX) {
                    j = 0;
                    i++;
                }
            }
            correct[0] += corr;
        });
        
    });
    return ret;
}

#if FPGA_EMULATOR
	ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
	ext::intel::fpga_selector d_selector;
#else
	default_selector d_selector;
#endif

void initialize(float *bases, float *classes, int numFeats, int numClass, int numDim) {

    std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<> d{0.0, 1.0};
    for(int i = 0; i < numFeats * numDim; i++) {
        bases[i] = d(gen);
    }
    for(int i = 0; i < numClass * numDim; i++) {
        classes[i] = 0.0;
    }
}

void shuffle(float *data, int *labels, int size, int numFeats) {

        srand(time(NULL));
        for(int i = 0; i < size; i++) {
            int res = rand() % size;
            for(int j = 0; j < numFeats; j++) {
                float tempData = data[i * numFeats + j];
                data[i * numFeats + j] = data[res * numFeats + j];
                data[res * numFeats + j] = tempData;
            }
            int tempLabel = labels[i];
            
            labels[i] = labels[res];
            
            labels[res] = tempLabel;
        }
}

std::vector<int> regen(float *classes, int numDim, int numClasses, int numToRegen) {
    std::vector<float> normed = normalizeClasses(classes, numClasses, numDim);
    std::vector<float>variances(numDim);
	for(int i = 0; i < numDim; i ++) {
		float mean = 0.0;
		for(int j = 0; j < numClasses; j++) {
			mean += normed[j * numDim + i];
		}
		mean = mean / (float) numClasses;
		float var = 0.0;
		for(int j = 0; j < numClasses; j++) {
			var += (normed[j * numDim + i] - mean) * (normed[j * numDim + i] - mean);
		}
		var = var / (float) numClasses;
		variances[i] = var;
	}
	std::vector<int> indeces(numDim);
	std::iota(indeces.begin(), indeces.end(), 0);
	std::stable_sort(indeces.begin(), indeces.end(), [&variances](int i1, int i2) {return variances[i1] < variances[i2];});
    return indeces;
}

void update(queue &q, float *bases, float *classes, int* indeces, float *rands) {
    std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<> d{0.0, 1.0};
    std::vector<float> randomVals(DMAX * NFEATMAX, 0.0);
    for(int i = 0; i < (PERC_UPDATE * DMAX) * NFEATMAX; i++) {
         randomVals[i] = d(gen);
    }
    
    q.memcpy(rands, randomVals.data(), sizeof(float) * randomVals.size()).wait();
    q.submit([&](handler &h){
		h.template single_task([=](){
            int upper = PERC_UPDATE * DMAX;
           // PRINTF("%d\n", upper);
           // #pragma unroll
            int index[DMAX];
            #pragma unroll 30
            for(int i = 0; i < DMAX; i++) {
                index[i] = indeces[i];
            }

            for(int j = 0; j < upper; j++) {
                #pragma unroll
                for(int k = 0; k < NCLASSMAX; k++) {
//                     if(k * D + indeces[j] >= D * NCLASS) {
//                         PRINTF("%d:\t%d\t%d\t%d\n", j, k, indeces[j], k * D + indeces[j]);
//                     }
                    classes[k * DMAX + index[j]] = 0.0;
                }
            }
            int count = 0;
            for(int j = 0; j < upper; j++) {
                int ind = index[j];
                //[[intel::ivdep]]
                #pragma unroll 16
                for(int k = 0; k < NFEATMAX; k++) {
                    bases[ind * NFEATMAX + k] = rands[count];
                    count++;
                }
            }
		});
	}).wait();


   
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
    sycl::property_list q_prop{sycl::property::queue::in_order()};
    queue q(d_selector, q_prop);
    
    int numFeats, numClasses, numDim, batch;
    
    numFeats = 784;
    numClasses = 10;
    numDim = 2000;
    batch = MAXBATCH;
    
    char* trainFile = strdup("mnist_train.choir_dat");
	Data train = readData(trainFile, q);

    int numData = train.data.size();
    std::cout << numData << "\n";
    
    
    numBatches = (numData / batch) + 1;
    
    float *bases = static_cast<float *>(malloc(sizeof(float) * numFeats * numDim));
    float *classes = static_cast<float *>(malloc(sizeof(float) * numClasses * numDim));
    float *data = static_cast<float *>(malloc(sizeof(float) * batch * numBatches * numFeats));
    int *labels = static_cast<int *>(malloc(sizeof(int) * batch * numBatches));
    //malloc_stats();
    initialize(bases, classes, numFeats, numClasses, numDim);
    for(int i = 0; i < numData; i++) {
        for(int j = 0; j < numFeats; j++) {
            data[i * numFeats + j] = train.data[i][j];
        }
    }
    q.memcpy(labels, train.labels, sizeof(float) * numData).wait();
    
    float *deviceBases = malloc_device<float>(numFeats * numDim, q);
    float *deviceClasses = malloc_device<float>(numClasses * numDim, q);
    float *deviceData = malloc_device<float>(batch * numBatches * numFeats, q);
    float *intermeds1 = malloc_device<float>(batch * numBatches * numDim, q);
    float *deviceRands = malloc_device<float>(numDim * numFeats, q);
    int *deviceIndeces = malloc_device<int>(numDim, q);
    int *deviceLabels = malloc_device<int>(batch * numBatches, q);
    int *correct = malloc_device<int>(1, q);
        //malloc_stats();

//     float *intermeds2 = malloc_device<float>(batch * numDim, q);
//     int *preds = malloc_device<int>(batch * numBatches * numClasses, q);
    
        //malloc_stats();
    q.memcpy(deviceBases, bases, sizeof(float) * numFeats * numDim).wait();
    q.memcpy(deviceClasses, classes, sizeof(float) * numClasses * numDim).wait();
    auto start = high_resolution_clock::now();
    //shuffle(data, labels, numData, numFeats);
    for(int i = 0; i < NUMREGEN; i++) {
        q.memcpy(deviceData, data, sizeof(float) * batch * numBatches * numFeats).wait();
        q.memcpy(deviceLabels, labels, sizeof(int) * batch * numBatches).wait();
        int startPtr = 0;
        //auto startEncode = high_resolution_clock::now();
        for(int j = 0; j < numBatches; j++) {
            encode(q, deviceBases, deviceData, intermeds1, startPtr, numDim, numFeats);
            startPtr += MAXBATCH;
            
        }
        shuffle(data, labels, numData, numFeats);
        q.wait();
        //auto endEncode = high_resolution_clock::now();
        //printf("ENCODE TIME: %f\n", elapsedTime(startEncode, endEncode));
        for(int j = 0; j < INTERIM; j++) {
            int localCorrect = 0;
            q.memcpy(correct, &localCorrect, sizeof(int)).wait();
            //auto startClassify = high_resolution_clock::now();
            int cur = 0;
            for(int k = 0; k < numBatches; k++) { 
                classify(q, deviceClasses, intermeds1, deviceLabels, cur, MIN(batch, numData - cur), numDim, numClasses, correct);
                cur += MAXBATCH;
                q.wait();
            }
            //auto endClassify = high_resolution_clock::now();
            //printf("CLASSIFY TIME: %f\n", elapsedTime(startClassify, endClassify));
            q.memcpy(&localCorrect, correct, sizeof(int)).wait();
            double accuracy = (double) localCorrect / (double) numData;
            //std::cout << accuracy << "\n";
            if(localCorrect == numData) {
                auto end = high_resolution_clock::now();
                printf("TIME: %f\n", elapsedTime(start, end));
                q.memcpy(bases, deviceBases, sizeof(float) * numFeats * numDim).wait();
                q.memcpy(classes, deviceClasses, sizeof(float) * numClasses * numDim).wait();
                char* testFile = strdup("mnist_test.choir_dat");
                Data test = readData(testFile, q);
                numData = test.data.size();
                q.memcpy(labels, test.labels, sizeof(int) * numData).wait();
                localCorrect = infer(bases, classes, test.data, labels, numFeats, numDim, numClasses);
                double accuracy = (double) localCorrect / (double) numData;
                std::cout << "\nACCURACY: " << accuracy << "\n\n";
                sycl::free(deviceBases, q);
                sycl::free(deviceClasses, q);
                sycl::free(deviceData, q);
                sycl::free(intermeds1, q);
                sycl::free(deviceIndeces, q);
                sycl::free(deviceRands, q);
                sycl::free(correct, q);
                sycl::free(deviceLabels, q);
                sycl::free(train.labels, q);
                sycl::free(test.labels, q);
                free(bases);
                free(classes);
                free(data);
                free(labels);
                return 0;
            }
        }
        //std::cout << "* * * * * * * * * *\n";
        //auto startRegen = high_resolution_clock::now();
        q.memcpy(classes, deviceClasses, sizeof(float) * numClasses * numDim).wait();
        std::vector<int> indeces = regen(classes, numDim, numClasses, 0.2 * numDim); 
        q.memcpy(deviceIndeces, indeces.data(), sizeof(float) * (int)(0.2 * numDim)).wait();
        update(q, deviceBases, deviceClasses, deviceIndeces, deviceRands);
        //auto endRegen = high_resolution_clock::now();
        //printf("REGEN TIME: %f\n", elapsedTime(startRegen, endRegen));
    }
    auto end = high_resolution_clock::now();
    printf("DID NOT CONVERGE\n");
    printf("TIME: %f\n", elapsedTime(start, end));
    sycl::free(deviceBases, q);
    sycl::free(deviceClasses, q);
    sycl::free(deviceData, q);
    sycl::free(intermeds1, q);
    sycl::free(correct, q);
    sycl::free(deviceLabels, q);
    sycl::free(deviceIndeces, q);
    sycl::free(deviceRands, q);
    sycl::free(train.labels, q);
    free(bases);
    free(classes);
    free(data);
    free(labels);
    return 0;
}