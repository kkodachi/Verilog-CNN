#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Declare the kernel function from convolution.cu
void launchConvolutionKernel(float* input, float* kernel, float* output, 
                             int inputWidth, int inputHeight, 
                             int kernelWidth, int kernelHeight);

int main() {
    auto start = std::chrono::high_resolution_clock::now(); 
    // Image and kernel dimensions
    const int inputWidth = 5, inputHeight = 5;
    const int kernelWidth = 3, kernelHeight = 3;

    const int outputWidth = inputWidth - kernelWidth + 1;
    const int outputHeight = inputHeight - kernelHeight + 1;

    // Allocate and initialize host memory
    float hostInput[inputWidth * inputHeight] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    float hostKernel[kernelWidth * kernelHeight] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    float hostOutput[outputWidth * outputHeight] = {0};

    // Allocate device memory
    float *devInput, *devKernel, *devOutput;
    cudaMalloc(&devInput, inputWidth * inputHeight * sizeof(float));
    cudaMalloc(&devKernel, kernelWidth * kernelHeight * sizeof(float));
    cudaMalloc(&devOutput, outputWidth * outputHeight * sizeof(float));

    // Copy data to device
    cudaMemcpy(devInput, hostInput, inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devKernel, hostKernel, kernelWidth * kernelHeight * sizeof(float), cudaMemcpyHostToDevice);

    
    // Launch the kernel
    launchConvolutionKernel(devInput, devKernel, devOutput, inputWidth, inputHeight, kernelWidth, kernelHeight);
    cudaDeviceSynchronize();
    

    // Copy result back to host
    cudaMemcpy(hostOutput, devOutput, outputWidth * outputHeight * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    // std::cout << "Convolution Output:\n";
    // for (int y = 0; y < outputHeight; ++y) {
    //     for (int x = 0; x < outputWidth; ++x) {
    //         std::cout << hostOutput[y * outputWidth + x] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Free device memory
    cudaFree(devInput);
    cudaFree(devKernel);
    cudaFree(devOutput);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "CUDA conv2d elapsed time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
