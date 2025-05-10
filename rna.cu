#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void gpu_warmup()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 1024)
    {
        float x = tid * 0.5f;
        x = sqrtf(x);
    }
}

__global__ void dnaToRna(char* dna, long N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
    {
        return;
    }

    dna[index] = dna[index] + ((dna[index] == 'T') * ('U' - 'T'));
}

int main()
{
    srand(time(NULL));

    long dnaLength = 1e9;
    size_t size = sizeof(char) * dnaLength;

    dim3 block (1024);
    dim3 grid ((dnaLength + block.x - 1) / block.x);

    char* h_dna = (char*) malloc(size);

    for (int i = 0; i < dnaLength; i++)
    {
        int decision = rand() % 4;

        if (decision == 0)
        {
            h_dna[i] = 'A';
        }
        else if (decision == 1)
        {
            h_dna[i] = 'C';
        }
        else if (decision == 2)
        {
            h_dna[i] = 'G';
        }
        else
        {
            h_dna[i] = 'T';
        }
    }

    char* d_dna;
    cudaMalloc((void**) &d_dna, size);

    cudaMemcpy(d_dna, h_dna, size, cudaMemcpyHostToDevice);

    // warmup
    gpu_warmup<<<128, 128>>>();

    cudaDeviceSynchronize();

    // actual
    dnaToRna<<<grid, block>>>(d_dna, dnaLength);

    cudaDeviceSynchronize();

    cudaMemcpy(h_dna, d_dna, size, cudaMemcpyDeviceToHost);

    printf("First 100 elements of rna \n");

    for (int i = 0; i < 100; i++)
        printf("%c", h_dna[i]);

    printf("\n");

    cudaFree(d_dna);

    free(h_dna);

    return 0;
}
