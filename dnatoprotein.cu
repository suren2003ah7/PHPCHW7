#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__constant__ char codonMap[2048];

__global__ void proteinFromDnaExtractor(char* rna, char* protein, long N, int* stopIndex)
{
    long idx = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
    long proteinIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    char result = codonMap[1 * rna[idx] + 4 * rna[idx + 1] + 16 * rna[idx + 2]];
    __syncthreads();

    if (result == '_')
    {
        atomicMin(stopIndex, proteinIdx);
    }

    __syncthreads();

    protein[proteinIdx] = result;
}

int main()
{
    srand(time(NULL));

    int rnaLength = 1000002; // multiple of 3
    int h_stopIndex = rnaLength;
    size_t size = sizeof(char) * rnaLength;

    dim3 block (32);
    dim3 grid ((rnaLength / 3 + block.x - 1) / block.x);

    char* h_rna = (char*) malloc(size);

    char* h_protein = (char*) malloc(sizeof(char) * rnaLength / 3);

    for (int i = 0; i < rnaLength; i++)
    {
        int decision = rand() % 4;

        if (decision == 0)
        {
            h_rna[i] = 'A';
        }
        else if (decision == 1)
        {
            h_rna[i] = 'C';
        }
        else if (decision == 2)
        {
            h_rna[i] = 'G';
        }
        else
        {
            h_rna[i] = 'U';
        }
    }

    char h_codonMap[2048] = {0};

    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'U'] = 'F';
    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'C'] = 'F';
    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'A'] = 'L';
    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'G'] = 'L';

    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'U'] = 'S';
    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'C'] = 'S';
    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'A'] = 'S';
    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'G'] = 'S';

    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'U'] = 'Y';
    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'C'] = 'Y';
    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'A'] = '_';
    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'G'] = '_';

    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'U'] = 'C';
    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'C'] = 'C';
    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'A'] = '_';
    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'G'] = 'W';

    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'U'] = 'L';
    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'C'] = 'L';
    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'A'] = 'L';
    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'G'] = 'L';

    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'U'] = 'P';
    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'C'] = 'P';
    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'A'] = 'P';
    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'G'] = 'P';

    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'U'] = 'H';
    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'C'] = 'H';
    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'A'] = 'Q';
    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'G'] = 'Q';

    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'U'] = 'R';
    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'C'] = 'R';
    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'A'] = 'R';
    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'G'] = 'R';

    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'U'] = 'I';
    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'C'] = 'I';
    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'A'] = 'I';
    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'G'] = 'M';

    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'U'] = 'T';
    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'C'] = 'T';
    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'A'] = 'T';
    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'G'] = 'T';

    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'U'] = 'N';
    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'C'] = 'N';
    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'A'] = 'K';
    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'G'] = 'K';

    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'U'] = 'S';
    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'C'] = 'S';
    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'A'] = 'R';
    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'G'] = 'R';

    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'U'] = 'V';
    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'C'] = 'V';
    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'A'] = 'V';
    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'G'] = 'V';

    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'U'] = 'A';
    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'C'] = 'A';
    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'A'] = 'A';
    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'G'] = 'A';

    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'U'] = 'D';
    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'C'] = 'D';
    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'A'] = 'E';
    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'G'] = 'E';

    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'U'] = 'G';
    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'C'] = 'G';
    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'A'] = 'G';
    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'G'] = 'G';

    cudaMemcpyToSymbol(codonMap, &h_codonMap, sizeof(char) * 2048);

    char* d_rna;
    cudaMalloc((void**) &d_rna, size);

    cudaMemcpy(d_rna, h_rna, size, cudaMemcpyHostToDevice);

    char* d_protein;
    cudaMalloc((void**) &d_protein, sizeof(char) * rnaLength / 3);

    cudaMemcpy(d_protein, h_protein, sizeof(char) * rnaLength / 3, cudaMemcpyHostToDevice);

    int* d_stopIndex;
    cudaMalloc((void**) &d_stopIndex, sizeof(int));

    cudaMemcpy(d_stopIndex, &h_stopIndex, sizeof(int), cudaMemcpyHostToDevice);

    proteinFromDnaExtractor<<<grid, block>>>(d_rna, d_protein, rnaLength, d_stopIndex);

    cudaDeviceSynchronize();

    cudaMemcpy(h_protein, d_protein, sizeof(char) * rnaLength / 3, cudaMemcpyDeviceToHost);

    cudaMemcpy(&h_stopIndex, d_stopIndex, sizeof(int), cudaMemcpyDeviceToHost);

    printf("True protein length: %d\n", h_stopIndex);

    printf("Head of rna: ");
    for (int i = 0; i < 30; i++)
        printf("%c", h_rna[i]);

    printf("\n");

    printf("Head of protein: ");
    for (int i = 0; i < 10; i++)
        printf("%c", h_protein[i]);

    printf("\n");

    cudaFree(d_rna);
    cudaFree(d_protein);
    cudaFree(d_stopIndex);

    free(h_rna);
    free(h_protein);

    return 0;
}
