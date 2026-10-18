#include <iostream>
#include <cuda_runtime.h>

__global__ void square(float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n ) {
        out[idx] = in[idx] * in[idx];
    }
}

int main() {
    // how many squares we'll do
    int n = 1000;

    // allocate cpu memory
    float* cpu_input = (float *)malloc(n * sizeof(float));
    float* cpu_output = (float *)malloc(n * sizeof(float));

    // fill cpu input with data
    for( int i = 0; i < n; ++i ) {
        cpu_input[i] = float(i);
    }

    // allocate gpu memory
    float* gpu_input; float* gpu_output;
    cudaMalloc(&gpu_input, n * sizeof(float));
    cudaMalloc(&gpu_output, n * sizeof(float));

    // move cpu input into gpu input
    cudaMemcpy(gpu_input,cpu_input,n,cudaMemcpyHostToDevice);

    // define kernel dimensions and launch
    int threadsPerBlock = 256;
    int gridSize = (n - threadsPerBlock - 1) / threadsPerBlock;
    square<<<gridSize, threadsPerBlock>>>(gpu_input, gpu_output, n);

    // move gpu output into cpu input and print
    cudaMemcpy(cpu_output,gpu_output,n,cudaMemcpyDeviceToHost);
    for( int i = 0; i < 5; ++i ) {
        std::cout << cpu_output[i] << std::endl;
    }

    cudaFree(gpu_input); cudaFree(gpu_output);
    free(cpu_input); free(cpu_output);

}
