#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__constant__ float weights[25];

__global__ void calculateWeightOfProtein(char* protein, long N, double* result)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
    {
        return;
    }

    atomicAdd(result, weights[protein[idx] - 'A']);
}

int main()
{
    srand(time(NULL));

    long proteinLength = 1e9;
    size_t size = sizeof(char) * proteinLength;

    dim3 block (64);
    dim3 grid ((proteinLength + block.x - 1) / block.x);

    char* h_protein = (char*) malloc(size);

    int i = 0;
    while (i < proteinLength)
    {
        int random = (rand() % 25) + 65;
        if (random == 66 || random == 74 || random == 79 || random == 85 || random == 88)
        {
            continue;
        }

        h_protein[i] = (char) random;
        i++;
    }

    float h_weights[25] = {
        71.03711, 0.0, 103.00919, 115.02694, 129.04259,
        147.06841, 57.02146, 137.05891, 113.08406, 0.0,
        128.09496, 113.08406, 131.04049, 114.04293, 0.0,
        97.05276, 128.05858, 156.10111, 87.03203, 101.04768,
        0.0, 99.06841, 186.07931, 0.0, 163.06333
    };

    double* h_result = (double*) malloc(sizeof(double));
    *h_result = 0;

    double* d_result;
    cudaMalloc((void**) &d_result, sizeof(double));

    cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(weights, &h_weights, sizeof(float) * 25);

    char* d_protein;
    cudaMalloc((void**) &d_protein, size);

    cudaMemcpy(d_protein, h_protein, size, cudaMemcpyHostToDevice);

    calculateWeightOfProtein<<<grid, block>>>(d_protein, proteinLength, d_result);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Result: %lf \n", *h_result);

    free(h_protein);
    free(h_result);

    cudaFree(d_protein);
    cudaFree(d_result);

    return 0;
}
