#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

cudaError_t softmaxWithCuda(float* input, float* output, int width, int height);

__global__ void softmaxkernel(float* input, float* output, int width, int height) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int index = row * width + col;

    if (row < height && col < width) {
        __shared__ float shared_value[1024];
        __shared__ float shared_sum;

        float exp_value = expf(input[index]);
        shared_value[col] = exp_value;

        __syncthreads();

        if (col == 0) {
            float sum = 0.0f;
            for (int i = 0; i < width; ++i) {
                sum += shared_value[i];
            }
            shared_sum = sum;
        }

        __syncthreads();

        output[index] = shared_value[col] / shared_sum;
    }
}

cudaError_t softmaxWithCuda(float* input, float* output, int width, int height) {
    float* dev_input = nullptr;
    float* dev_output = nullptr;
    cudaError_t cudaStatus;

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_input, width * height * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_input failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, width * height * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_output failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy input data from host to device
    cudaStatus = cudaMemcpy(dev_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy input failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Launch kernel
    dim3 blocknums(height);    // Each block processes one row
    dim3 threadsperblock(width); // Each thread processes one column
    softmaxkernel << <blocknums, threadsperblock >> > (dev_input, dev_output, width, height);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy output from device to host
    cudaStatus = cudaMemcpy(output, dev_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy output failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    cudaFree(dev_input);
    cudaFree(dev_output);
    return cudaStatus;
}
