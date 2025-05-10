#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__constant__ float c;

__global__ void earthquakePropagation(float* prev, float* curr, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    extern __shared__ float next[];

    if (idx >= 3 && idx <= N - 4)
    {
        next[threadIdx.x] = 2 * curr[idx] - prev[idx] + c * (curr[idx-3] - 6*curr[idx-2] + 15*curr[idx-1] - 20*curr[idx] + 15*curr[idx+1] - 6*curr[idx+2] + curr[idx+3]);
    }
    __syncthreads();

    if (idx >= 3 && idx <= N - 4) 
    {
        prev[idx] = curr[idx];
        curr[idx] = next[threadIdx.x];
    }
}

int main()
{
    srand(time(NULL));

    int length = 100;
    int iterCount = 200;
    size_t size = sizeof(float) * length;

    dim3 block (32);
    dim3 grid ((length + block.x - 1) / block.x);

    float h_c = 0.01f;

    cudaMemcpyToSymbol(c, &h_c, sizeof(float));

    float* h_curr = (float*) malloc(size);
    float* h_prev = (float*) malloc(size);

    for (int i = 0; i < length; i++)
    {
        h_curr[i] = 0;
        h_prev[i] = 0;
    }
    
    h_curr[length / 2] = 100;
    h_prev[length / 2] = 100;

    for (int i = 0; i < length; i++)
        printf("%f ", h_curr[i]);
    
    printf("\n");

    float* d_curr;
    float* d_prev;

    cudaMalloc((void**) &d_curr, size);
    cudaMalloc((void**) &d_prev, size);

    cudaMemcpy(d_curr, h_curr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev, h_prev, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < iterCount; i++)
    {
        earthquakePropagation<<<grid, block, block.x * sizeof(float)>>>(d_prev, d_curr, length);
        cudaDeviceSynchronize();

        cudaMemcpy(h_curr, d_curr, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < length; i++)
            printf("%f ", h_curr[i]);
    
        printf("\n");
    }

    free(h_prev);
    free(h_curr);

    cudaFree(d_prev);
    cudaFree(d_curr);

    return 0;
}
