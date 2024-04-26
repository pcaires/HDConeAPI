// Copyright (c) 2018 NSF Center for Space, High-performance, and Resilient Computing (SHREC)
// University of Pittsburgh. All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted provided
// that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, 
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <cmath>
#include <fstream>
#include <stdint.h>
#include <arpa/inet.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#define TIME (std::chrono::high_resolution_clock::now())
#define FOR(i,n) for(int i = 0; i < n; i++)
#define FORX(it,iterable) for(auto &it : iterable)

using namespace cl::sycl;

//typedefs
typedef std::vector<std::vector<float>> fmat;
typedef std::vector<float> fvec;
typedef std::vector<int> ivec;
typedef cl::sycl::buffer<int> ibuf;
typedef cl::sycl::buffer<float> fbuf;
typedef std::chrono::high_resolution_clock::time_point timep;

typedef struct{
    fmat data;
    ivec labels;
    int32_t numFeatures;
    int32_t numClasses;
} Data;


//global variables
static const int32_t ndims = 2000;
static const float alpha = 0.037;
static int32_t nfeatures, nclasses, work_group_size;
static fvec basisv, classesv;
static fbuf *basis_bufp = nullptr, *classes_bufp = nullptr;
static queue *q = nullptr;
static fvec accuracies, inference_times, train_times, runtimes;

static double test_times[2];
static timep tstart, tend;

default_selector d_selector;

static auto e_handler = [](exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
      std::cout << "Failure" << std::endl;
      std::terminate();
    }
  }
};


double elapsedTime(timep start, timep end) {
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
  return time_span.count();
}

fvec normalize(fvec input) {
	float inner = std::inner_product(input.begin(), input.end(), input.begin(), 0.0);
	float norm = std::sqrt(inner);
	for(int i = 0; i < input.size(); i++) {
		input[i] = input[i] / norm;
	}
	return input;
}

fvec m2v(fmat in){
	fvec out(in.size() * in[0].size(), 0.0);
	for(int i = 0; i < in.size(); i++) {
		for(int j = 0; j < in[0].size(); j++) {
			out[i * in[0].size() + j] = in[i][j];
		}
	}
    return out;
}

void encode(fbuf &data_buf, fbuf &out_buf, const size_t ndata){

    if (out_buf.get_range()[0] != ndata*ndims)
        throw std::runtime_error("Dimesions do not match " + std::to_string(out_buf.get_range()[0]/ndims) + " " + std::to_string(data_buf.get_range()[0]/(nfeatures)));

    if (data_buf.get_range()[0] != ndata*nfeatures)
        throw std::invalid_argument("data buff size does not match expected size: " + std::to_string(ndata*nfeatures) + " got instead:" + std::to_string(data_buf.get_range()[0]));

    q->submit([&](handler &cgh) {
        auto C_acc = out_buf.get_access(cgh,write_only,no_init);
        auto A_acc = data_buf.get_access(cgh,read_only);
        auto B_acc = basis_bufp->get_access(cgh,read_only);

        cgh.parallel_for(
            range<2>{ndata, ndims}, [=](id<2> item) {
                const int dt_point = item[0];
                const int dim = item[1];
                float result = 0;
                for (int i = 0; i < nfeatures; i++) 
                    result += A_acc[dt_point * nfeatures + i] * B_acc[i * ndims + dim];
    
                C_acc[dt_point * ndims + dim] = 0;//cl::sycl::cos(result);
            });
    });
}

Data readData(char *filename) {
    std::ifstream testFile(filename, std::ifstream::binary);
    char *holder = (char *)malloc(4 * sizeof(char));
    testFile.read(holder, 4 * sizeof(char));
    int32_t numFeatures;
    memcpy(&numFeatures, &holder[0], sizeof(numFeatures));
    testFile.read(holder, 4 * sizeof(char));
    int32_t numClasses;
    memcpy(&numClasses, &holder[0], sizeof(numClasses));
    fmat testData;
    ivec testLabels;
    while(testFile.good()) {
        fvec vect(numFeatures, 0.0);
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
    Data ret;
    ret.data = testData;
    ret.labels = testLabels;
    ret.numClasses = numClasses;
    ret.numFeatures = numFeatures;
    return ret;
}


int cpuFit(fbuf &dataBuf, ivec &labels, fvec &classes) {
    host_accessor data(dataBuf,read_only);
    int correct = 0;
    ivec guesses(labels.size(), 0);
    FOR(i, labels.size()) {
        float max = 0.0f;
        int guess = 0;
        FOR(j, nclasses){
            float sum = 0.0f;
            for(int k = 0; k < ndims; k++) {
                sum += classes[j * ndims + k] * data[i * ndims + k];
            }
            if(sum > max) {
                max = sum;
                guess = j;
            }
        }
        guesses[i] = guess;
        if(guess == labels[i]) {
            correct ++;
        }
        else {
            FOR(j, ndims){
                classes[guess * ndims + j] -= 0.037 * data[i * ndims + j];
                classes[labels[i] * ndims + j] += 0.037 * data[i * ndims + j];
            }
        }
    }
    return correct;
}

void testNN2(fbuf &data, ivec &labels, int &correct){
    int ndata = labels.size();

    ibuf label_buf(labels);
    ivec correct_vec(ndata, 0);
    ibuf correct_buf(correct_vec);

    buffer<int, 1> correct_buf_single(&correct, 1);
    q->submit([&](auto &h){
        accessor label_a(label_buf, h, read_only);
        accessor data_a(data, h, read_only);
        accessor class_a(*classes_bufp, h, read_only);
        accessor correct_a(correct_buf, h, read_write);

        h.parallel_for(range(ndata), [=](auto i){
            float max = -1.0;
            int guess = -1;
            int index = i[0];
            FOR(j, nclasses){
                float cum = 0;
                FOR(k, ndims){
                    cum += data_a[index*ndims+k] * class_a[j*ndims+k];
                }
                if(cum > max){
                    max = cum;
                    guess = j;
                }
            }
            if(guess == label_a[index]){
                correct_a[index] = 1;
            }
        });
    });
    q->wait();

    q->submit([&](auto &h){
        accessor correct_a(correct_buf, h, read_only);
        accessor correct_a_out(correct_buf_single, h, read_write);

        h.single_task([=](){
            FOR(i, ndata) correct_a_out[0] += correct_a[i];
        });
    });
    q->wait();
}

/**
 * @brief A function that calculates the variance of each dimension and 
 * returns the indexes in ascending order of variance.
 *
 * 
 * @return ivec (lowest variance dim index -> highest variance dim index)
 */
ivec rankDims(){
    fvec variances(ndims);
    fbuf *var_buf = new fbuf(variances);
    ivec sorted_indexes(ndims);
    std::iota(sorted_indexes.begin(), sorted_indexes.end(), 0);
    ibuf sorted_indexes_buf(sorted_indexes);

    // SYCL kernel to calculate vaiances
    q->submit([&](auto &h){
        accessor class_a(*classes_bufp, h, read_only);
        accessor var_a(*var_buf, h, write_only);

       
        int nclasses_ = nclasses;

        h.parallel_for(range(ndims), [=](auto index){
            int d = index[0];
            float mean=(float)0, sum=(float)0;
            FOR(c, nclasses_){
                mean += class_a[c*ndims+d];
            }
            mean *= (float)1 / (float)nclasses_;
            FOR(c, nclasses_){
                float temp = class_a[c*ndims+d] - mean;
                sum += temp*temp;
            }
            var_a[d] = sum;
        });
    });
    q->wait();
    delete var_buf; var_buf = nullptr;

    //sort indeces in ascending order
    std::sort(sorted_indexes.begin(), sorted_indexes.end(),
        [&](int i, int j){ return (variances[i] < variances[j]); });
        
    return sorted_indexes;
}

/**
 * @brief Update Classes and Basis Vectors
 * 
 * @param dim_ranks Highes variance dimesion indexes
 */
void updateClassesAndBasis(const ivec &dim_ranks){

    const int dim_loss = dim_ranks.size();
    
    srand (time(NULL));
    int seed = rand();
    host_accessor basis_a(*basis_bufp, read_write);

    int nfeatures_ = nfeatures;

    std::mt19937 rng(seed);
    std::normal_distribution<float> gen{0,1};

    FOR(i, nfeatures_)
        FOR(j, dim_loss)
            basis_a[i*ndims + dim_ranks[j]] = gen(rng);

    ibuf dim_ranks_buf(dim_ranks);

    q->submit([&](auto &h){
        accessor class_a(*classes_bufp, h, read_write);
        accessor dim_ranks_a(dim_ranks_buf, h, read_only);

        int nclasses_ = nclasses;

        h.parallel_for(range(nclasses_, dim_loss), [=](auto index){
            class_a[index[0]*ndims + dim_ranks_a[index[1]]] = (float)0;
        });
    });
    q->wait();
}

void trainAndTestWithRegen(){
    
    std::cout << "Enter function with dim: " << ndims << std::endl;
    q = new queue(d_selector, e_handler);
    std::cout << "Hardware: " << q->get_device().get_info<info::device::name>() << std::endl;
    work_group_size = q->get_device().get_info<info::device::max_work_group_size>();
    std::cout << work_group_size << std::endl;
    std::cout << "reading files... " << std::endl;
	char *testFile = strdup("data/mnist_test.choir_dat");
	Data test = readData(testFile);
	fvec test_data = m2v(test.data);
	ivec test_labels = test.labels;

	char* trainFile = strdup("data/mnist_train.choir_dat");
	Data train = readData(trainFile);
    fvec train_data = m2v(train.data);
	ivec train_labels = train.labels;
    std::cout << "success!" << std::endl;
    
	nclasses = test.numClasses;
	nfeatures = test.numFeatures;

    std::cout << "allocate basis and class (modified)" << std::endl;
    basisv = fvec(nfeatures*ndims);
    classesv = fvec(nclasses*ndims, 0.0);

    std::cout << "generate basis: " << nfeatures << " " << ndims << std::endl;
    //generate basis
    tstart = TIME;
    srand (time(NULL));
    int seed = rand();


    std::mt19937 rng(seed);
    std::normal_distribution<float> gen;

    FOR(i, nfeatures*ndims)
        basisv[i] = gen(rng);

    tend = TIME;
    std::cout << "generate basis time: " << elapsedTime(tstart, tend) << std::endl;
    
    basis_bufp = new fbuf(basisv);
    classes_bufp = new fbuf(classesv);

    const int ndata_train = train_labels.size();
    fvec train_data_encoded(ndata_train * ndims, 0);
    fbuf train_data_buf(train_data);
    fbuf train_data_encoded_buf(train_data_encoded);
    const int ndata_test = test_labels.size();
    fvec test_data_encoded(ndata_test * ndims, 0);
    fbuf test_data_buf(test_data);
    fbuf test_data_encoded_buf(test_data_encoded);
    tstart = TIME;
    encode(train_data_buf, train_data_encoded_buf, ndata_train);
    std::cout << "encode training data: " << elapsedTime(tstart, TIME) << std::endl;
    tstart = TIME;
    encode(test_data_buf, test_data_encoded_buf, ndata_test);
    std::cout << "encode testing data: " << elapsedTime(tstart, TIME) << std::endl;

	int regen = 0;//200; // number of regen iterations
	int interim = 5; // number of iters between regens (retraining iterations adapthd)
	double percent_drop = 0.2; // how many dims to regen per regen iter
    
    //ibuf test_label_buf
    fvec testClasses(ndims * nclasses, 0.0);
    tstart = TIME;
    double fit_time = 0;
    double test_time = 0;
    double recalc_time = 0;
    test_times[0] = 0;
    test_times[1] = 0;
    FOR(i, regen){
        //cout << "regen #" << i << ", dims=" << ndims << "   ";
        int test_acc = 0, train_acc = 0;
        q->wait();
        FOR(j, interim){
            test_acc = 0, train_acc = 0;

            auto t1 = TIME;

            // === This cpufit funcion does the adapthd retraining ===
            train_acc = cpuFit(train_data_encoded_buf, train_labels, testClasses);
            
            // Timing
            fit_time += elapsedTime(t1, TIME);
            t1 = TIME;
            test_time += elapsedTime(t1, TIME);

            // Output accuracy
            std::cout << (float) train_acc / (float) ndata_train << "\n";

            // What is the purpose of this 'if'? test_acc is always 0 and ndata_test is constant
            //if (test_acc == ndata_test){
            //    i = regen; //super break
            //    break;
            //}
        }
        q->wait();

        auto t2 = TIME;
        std::cout << test_acc << "   " << train_acc << std::endl;

        // how many dimensions to drop/regen
        int dim_loss = (float)ndims * (float)percent_drop;
        
        // Get the ranked dimensions (low var -> high var)
        ivec dim_ranks = rankDims();

        // Resize the ranked dimensions to the n lowest variance
        dim_ranks.resize(dim_loss);

        updateClassesAndBasis(dim_ranks);
        encode(train_data_buf, train_data_encoded_buf, ndata_train);
        encode(test_data_buf, test_data_encoded_buf, ndata_test);
        recalc_time += elapsedTime(t2, TIME);
    }
    auto train_time = elapsedTime(tstart, TIME);
    std::cout << std::endl;
    std::cout << "training time: " << train_time << std::endl;
    std::cout << "     fit time: " << fit_time << std::endl;
    std::cout << "    test time: " << test_time << std::endl;
    std::cout << "  recalc time: " << recalc_time << std::endl;
    std::cout << std::endl;

    tstart = TIME;
    int accuracy = 0;
    testNN2(test_data_encoded_buf, test_labels, accuracy);
    accuracies.push_back((float)accuracy / (float)ndata_test);
    train_times.push_back((float)train_time);
    double inference_time = elapsedTime(tstart, TIME);
    std::cout << "Inference time: " << inference_time << std::endl;
    std::cout << "Accuracy: " << accuracy << " " << accuracies[accuracies.size()-1] << std::endl;
}

void testInferenceBaseline(){
    
    std::cout << "Enter function with dim: " << ndims << std::endl;
    q = new queue(d_selector, e_handler);
    std::cout << "Hardware: " << q->get_device().get_info<info::device::name>() << std::endl;
    work_group_size = q->get_device().get_info<info::device::max_work_group_size>();

    std::cout << "reading files... " << std::endl;
	char *testFile = strdup("data/mnist_test.choir_dat");
	Data test = readData(testFile);
	fvec test_data = m2v(test.data);
	ivec test_labels = test.labels;

	char* trainFile = strdup("data/mnist_train.choir_dat");
	Data train = readData(trainFile);
    fvec train_data = m2v(train.data);
	ivec train_labels = train.labels;
    std::cout << "success!" << std::endl;
    
	nclasses = test.numClasses;
	nfeatures = test.numFeatures;

    std::cout << "classes, features: " << nclasses << " " << nfeatures << std::endl;

    std::cout << "allocate basis and class bufs" << std::endl;
    basisv = fvec(nfeatures*ndims, 0.0);
    basis_bufp = new fbuf(basisv);
    classesv = fvec(nclasses*ndims, 0.0);
    classes_bufp = new fbuf(classesv);

    std::cout << "generate basis: " << nfeatures << " " << ndims << std::endl;
    //generate basis
    tstart = TIME;
    host_accessor basis_a(*basis_bufp, write_only, no_init);

    srand(time(NULL));
    int seed = rand();

    std::mt19937 rng(seed);
    std::normal_distribution<float> gen{0,1};

    FOR(i, nfeatures*ndims)
        basis_a[i] = gen(rng);
    tend = TIME;
    
    int ndata_train = train_labels.size();

    int ndata_test = test_labels.size();

    std::vector<double> times(1);
    FOR(time_i, times.size()){
        int batches = 500;
        int inputs = 1;
        std::vector<fvec> test_data;
        std::vector<fbuf> test_datab;
        ivec garbage_labels(inputs);
        FOR(i,batches) {
            test_data.push_back(fvec(inputs * ndims));
            test_datab.push_back(fbuf(test_data[i]));
        }
        tstart = TIME;
        int acc;
        FOR(i,batches){
            testNN2(test_datab[i], garbage_labels, acc);
        }
        double timed = elapsedTime(tstart, TIME);
        times[time_i] = timed;
        std::cout << "run #" << time_i+1 << "   time: " << timed << std::endl;
    }
    double average_time = 0;
    FORX(x,times) average_time += x/10.0;
    std::cout << "average: " << average_time << std::endl;
}


int main(int argc, char **argv){
    std::cout << "Starting inference only: " << ndims << std::endl;
    //trainAndTestWithRegen();
    testInferenceBaseline();
    return 0;
}