#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void hammingDistanceCalculator(char* dna_1, char* dna_2, long N, int* distance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int localDistance;

    if (idx >= N)
        return;

    if (threadIdx.x == 0)
    {
        localDistance = 0;
    }

    int result = dna_1[idx] != dna_2[idx];

    atomicAdd(&localDistance, result);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(distance, localDistance);
    }
}

int main()
{
    srand(time(NULL));

    long dnaLength = 1e9;
    size_t size = sizeof(char) * dnaLength;

    dim3 block (1024);
    dim3 grid ((dnaLength + block.x - 1) / block.x);

    char* h_dna_1 = (char*) malloc(size);
    char* h_dna_2 = (char*) malloc(size);

    for (int i = 0; i < dnaLength; i++)
    {
        int decision = rand() % 4;

        if (decision == 0)
        {
            h_dna_1[i] = 'A';
        }
        else if (decision == 1)
        {
            h_dna_1[i] = 'C';
        }
        else if (decision == 2)
        {
            h_dna_1[i] = 'G';
        }
        else
        {
            h_dna_1[i] = 'T';
        }
    }

    for (int i = 0; i < dnaLength; i++)
    {
        int decision = rand() % 4;

        if (decision == 0)
        {
            h_dna_2[i] = 'A';
        }
        else if (decision == 1)
        {
            h_dna_2[i] = 'C';
        }
        else if (decision == 2)
        {
            h_dna_2[i] = 'G';
        }
        else
        {
            h_dna_2[i] = 'T';
        }
    }

    int* h_distance = (int*) malloc(sizeof(int));
    *h_distance = 0;

    char* d_dna_1;
    char* d_dna_2;

    cudaMalloc((void**) &d_dna_1, size);
    cudaMalloc((void**) &d_dna_2, size);

    cudaMemcpy(d_dna_1, h_dna_1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dna_2, h_dna_2, size, cudaMemcpyHostToDevice);

    int* d_distance;

    cudaMalloc((void**) &d_distance, sizeof(int));

    cudaMemcpy(d_distance, h_distance, sizeof(int), cudaMemcpyHostToDevice);

    hammingDistanceCalculator<<<grid, block>>>(d_dna_1, d_dna_2, dnaLength, d_distance);

    cudaDeviceSynchronize();

    cudaMemcpy(h_distance, d_distance, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Distance: %d\n", *h_distance);

    free(h_distance);
    free(h_dna_1);
    free(h_dna_2);

    cudaFree(d_dna_1);
    cudaFree(d_dna_2);
    cudaFree(d_distance);

    return 0;
}
