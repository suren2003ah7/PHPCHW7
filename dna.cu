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


__global__ void calculateNucleotideFrequency(char* dna, int* globalNucleotideFrequency, long N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
    {
        return;
    }

    __shared__ int nucleotideFrequencyTable[5];
    if (tid == 0)
    {
        for (int i = 0; i < 5; i++)
        {
            nucleotideFrequencyTable[i] = 0;
        }
    }
    __syncthreads();

    int resultIndex = dna[index] % 5;
    atomicAdd(&nucleotideFrequencyTable[resultIndex], 1);
    __syncthreads();

    if (tid == 0)
    {
        for (int i = 0; i < 5; i++)
        {
            atomicAdd(&globalNucleotideFrequency[i], nucleotideFrequencyTable[i]);
        }
    }
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

    int h_globalNucleotideFrequency[5] = {0};

    int* d_globalNucleotideFrequency;
    cudaMalloc((void**) &d_globalNucleotideFrequency, sizeof(int) * 5);

    cudaMemcpy(d_globalNucleotideFrequency, h_globalNucleotideFrequency, sizeof(int) * 5, cudaMemcpyHostToDevice);

    // warmup
    gpu_warmup<<<128, 128>>>();

    cudaDeviceSynchronize();

    // actual
    calculateNucleotideFrequency<<<grid, block>>>(d_dna, d_globalNucleotideFrequency, dnaLength);

    cudaDeviceSynchronize();

    cudaMemcpy(h_globalNucleotideFrequency, d_globalNucleotideFrequency, sizeof(int) * 5, cudaMemcpyDeviceToHost);

    printf("A: %d, C: %d, G: %d, T: %d\n", h_globalNucleotideFrequency[0], h_globalNucleotideFrequency[2], h_globalNucleotideFrequency[1], h_globalNucleotideFrequency[4]);

    cudaFree(d_globalNucleotideFrequency);
    cudaFree(d_dna);

    free(h_dna);

    return 0;
}
