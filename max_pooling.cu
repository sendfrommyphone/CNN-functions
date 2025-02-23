#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>

__global__ void maxpoolingkernel(float* input, float* output, int N, int C, int H, int W, int kernel_size, int stride) {
    int batchidx = blockIdx.x;  
    int channelidx = blockIdx.y;
    int out_h = threadIdx.x;  
    int out_w = threadIdx.y;  

    int start_h = out_h * stride;
    int start_w = out_w * stride;

    float max_value = -FLT_MAX; 

    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int input_h = start_h + kh;
            int input_w = start_w + kw;
            if (input_h < H && input_w < W) {
                int inputidx = (batchidx * C + channelidx) * H * W + input_h * W + input_w;
                max_value = fmaxf(max_value, input[inputidx]); 
            }
        }
    }


    int output_height = H / stride;
    int output_width = W / stride;
    int outputidx = (batchidx * C + channelidx) * output_height * output_width + out_h * output_width + out_w;
    output[outputidx] = max_value; 
}

cudaError_t maxpoolingWithCuda(float* input, float* output, int N, int C, int H, int W, int kernel_size, int stride) {
    float* dev_input;
    float* dev_output;
    cudaError_t cudaStatus;


    cudaStatus = cudaMalloc((void**)&dev_input, sizeof(float) * N * C * H * W);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_input\n");
        return cudaStatus;
    }

    int output_height = H / stride;
    int output_width = W / stride;
    cudaStatus = cudaMalloc((void**)&dev_output, sizeof(float) * N * C * output_height * output_width);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_output\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_input, input, sizeof(float) * N * C * H * W, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for input\n");
        return cudaStatus;
    }


    dim3 blocknum(N, C);
    dim3 threadsperblock(16, 16);

    maxpoolingkernel << <blocknum, threadsperblock >> > (dev_input, dev_output, N, C, H, W, kernel_size, stride);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "maxpooling failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(output, dev_output, sizeof(float) * N * C * output_height * output_width, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for output\n");
        return cudaStatus;
    }

    cudaFree(dev_input);
    cudaFree(dev_output);
    return cudaSuccess;
}


