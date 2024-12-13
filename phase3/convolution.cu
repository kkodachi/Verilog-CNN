#include <cuda_runtime.h>

// CUDA Kernel for 2D Convolution
__global__ void convKernel(float* input, float* kernel, float* output, 
                                  int inputWidth, int inputHeight, 
                                  int kernelWidth, int kernelHeight) {
    // calc thread ind
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    // if valid in
    if (outX < outputWidth && outY < outputHeight) {
        float sum = 0.0f;
        // each thread computes its portion of MM
        for (int ky = 0; ky < kernelHeight; ++ky) {
            for (int kx = 0; kx < kernelWidth; ++kx) {
                int inX = outX + kx;
                int inY = outY + ky;
                sum += input[inY * inputWidth + inX] * kernel[ky * kernelWidth + kx];
            }
        }

        output[outY * outputWidth + outX] = sum;
    }
}

// wrapper for launching  kernel
void startConvKernel(float* input, float* kernel, float* output, 
                             int inputWidth, int inputHeight, 
                             int kernelWidth, int kernelHeight) {
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);

    convKernel<<<gridSize, blockSize>>>(input, kernel, output, 
                                               inputWidth, inputHeight, 
                                               kernelWidth, kernelHeight);
    // wait for all threads to finish
    cudaDeviceSynchronize();
}
