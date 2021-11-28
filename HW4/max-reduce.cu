
#include <stdio.h>
#include <cuda.h>
#include <time.h>


#define MAX_CUDA_THREADS_PER_BLOCK 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Startup{
    int random_range = INT_MAX;
    int threads_per_block = MAX_CUDA_THREADS_PER_BLOCK;
} startup;

struct DataSet{
    int* values;
    int  size;
};

struct Result{
    int MaxValue;
    float KernelExecutionTime;
};

DataSet generateRandomDataSet(int size){
    DataSet data;
    data.size = size;
    data.values = (int*)malloc(sizeof(int)*data.size);

    for (int i = 0; i < data.size; i++)
        data.values[i] = (int)(rand()%startup.random_range);

    return data;
}

__global__ void Max_Sequential_Addressing_Shared(int* data, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                int lhs = sdata[threadIdx.x];
                int rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}


Result calculateMaxValue(DataSet data){
    int* device_data;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    gpuErrchk(cudaMalloc((void **)&device_data,  sizeof(int)*data.size));
    gpuErrchk(cudaMemcpy(device_data, data.values, sizeof(int)*data.size, cudaMemcpyHostToDevice));


    int threads_needed = data.size;
    cudaEventRecord(start);
    Max_Sequential_Addressing_Shared<<< threads_needed/ startup.threads_per_block + 1, startup.threads_per_block>>>(device_data, data.size);
    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    int max_value;
    gpuErrchk(cudaMemcpy(&max_value, device_data, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_data));

    Result r = {max_value, milliseconds}; // this might cause error
    return r;
}


void printDataSet(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%d, ", data.values[i]);
    printf("\n");
}


void benchmarkCSV(){
    /*Benchmark*/
    FILE *out_file = fopen("Results", "w");

    int size[] = {1, 5, 10, 15, 20};
    for (int i = 0; i<5; i++){
        int dataSize = size[i]*1000000;
        DataSet random = generateRandomDataSet(dataSize);
        Result r = calculateMaxValue(random);

        fprintf(out_file, "Data size: %d\n", dataSize);
        //fprintf(out_file, "Maximum value: %d\n", r.MaxValue);
        fprintf(out_file, "Execution time: %f\n----\n", r.KernelExecutionTime);

        printf("%d, ", dataSize);
        printf("%g, ", r.KernelExecutionTime);
        printf("\n");
        free(random.values);
    }
}


int main(int argc, char** argv){
    srand(time(nullptr));
    benchmarkCSV();
}

