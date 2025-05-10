#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__constant__ float decay_rate;

__global__ void chemicalDiffusion(float* data, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= 2 && idx <= N - 3)
    {
        data[idx] = (1 - decay_rate) * (0.1 * data[idx-2] + 0.2 * data[idx-1] + 0.4 * data[idx] + 0.2 * data[idx+1] + 0.1 * data[idx+2]);
    }
}

int main()
{
    srand(time(NULL));

    int length = 100;
    int iterCount = 200;
    size_t size = sizeof(float) * length;

    float h_decay_rate = 0.05;

    cudaMemcpyToSymbol(decay_rate, &h_decay_rate, sizeof(float));

    dim3 block (32);
    dim3 grid ((length + block.x - 1) / block.x);

    float* h_data = (float*) malloc(size);

    for (int i = 0; i < length; i++)
        h_data[i] = 0.0;
    
    h_data[length / 2] = 1.0;

    for (int i = 0; i < length; i++)
        printf("%f ", h_data[i]);
    
    printf("\n");

    float* d_data;
    cudaMalloc((void**) &d_data, size);

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < iterCount; i++)
    {
        chemicalDiffusion<<<grid, block>>>(d_data, length);

        cudaDeviceSynchronize();

        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < length; i++)
            printf("%f ", h_data[i]);
    
        printf("\n");
    }

    free(h_data);

    cudaFree(d_data);

    return 0;
}
