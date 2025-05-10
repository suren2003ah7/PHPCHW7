#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void gaussianBlur(float* data, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= 1 && idx <= N - 2)
    {
        data[idx] = 0.25 * data[idx - 1] + 0.5 * data[idx] + 0.25 * data[idx + 1];
    }
}

int main()
{
    srand(time(NULL));

    int length = 100;
    int iterCount = 5;
    size_t size = sizeof(float) * length;

    dim3 block (32);
    dim3 grid ((length + block.x - 1) / block.x);

    float* h_data = (float*) malloc(size);

    for (int i = 0; i < length; i++)
        h_data[i] = (float) ((float) rand() / (float) RAND_MAX) * 5;
    
    h_data[rand() % length] = 100;
    h_data[rand() % length] = 120;
    h_data[rand() % length] = 150;

    printf("Initial data\n");

    for (int i = 0; i < length; i++)
        printf("%f ", h_data[i]);
    
    printf("\n");

    float* d_data;
    cudaMalloc((void**) &d_data, size);

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < iterCount; i++)
    {
        gaussianBlur<<<grid, block>>>(d_data, length);

        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    printf("Output data after %d iterations \n", iterCount);

    for (int i = 0; i < length; i++)
        printf("%f ", h_data[i]);
    
    printf("\n");

    free(h_data);

    cudaFree(d_data);

    return 0;
}
