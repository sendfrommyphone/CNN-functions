#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "device_launch_parameters.h"

#define M_PI 3.1415926

__device__ float gelu(float x) {
	return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void gelukernel(float* input, float* output, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		output[idx] = gelu(input[idx]);
	}
}


cudaError_t geluWithCuda(float* input, float* output, int N) {
    float* dev_input;
    float* dev_output;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_input, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_input\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_output\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for input\n");
        return cudaStatus;
    }

    int blockSize = 256; 
    int numBlocks = (N + blockSize - 1) / blockSize;


    gelukernel << <numBlocks, blockSize >> > (dev_input, dev_output, N);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "applyGelu kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(output, dev_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for output\n");
        return cudaStatus;
    }

    // 释放设备内存
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaSuccess;
}

