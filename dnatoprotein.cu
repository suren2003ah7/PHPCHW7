#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__constant__ char codonMap[421];

__global__ void proteinFromDnaExtractor(char* rna, char* protein, long N, int* stopIndex)
{
    long idx = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
    long proteinIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    char result = codonMap[1 * rna[idx] + 4 * rna[idx + 1] + 16 * rna[idx + 2] - 1365];
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

    char h_codonMap[421] = {0};

    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'U' - 1365] = 'F';
    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'C' - 1365] = 'F';
    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'A' - 1365] = 'L';
    h_codonMap[1 * 'U' + 4 * 'U' + 16 * 'G' - 1365] = 'L';

    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'U' - 1365] = 'S';
    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'C' - 1365] = 'S';
    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'A' - 1365] = 'S';
    h_codonMap[1 * 'U' + 4 * 'C' + 16 * 'G' - 1365] = 'S';

    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'U' - 1365] = 'Y';
    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'C' - 1365] = 'Y';
    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'A' - 1365] = '_';
    h_codonMap[1 * 'U' + 4 * 'A' + 16 * 'G' - 1365] = '_';

    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'U' - 1365] = 'C';
    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'C' - 1365] = 'C';
    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'A' - 1365] = '_';
    h_codonMap[1 * 'U' + 4 * 'G' + 16 * 'G' - 1365] = 'W';

    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'U' - 1365] = 'L';
    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'C' - 1365] = 'L';
    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'A' - 1365] = 'L';
    h_codonMap[1 * 'C' + 4 * 'U' + 16 * 'G' - 1365] = 'L';

    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'U' - 1365] = 'P';
    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'C' - 1365] = 'P';
    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'A' - 1365] = 'P';
    h_codonMap[1 * 'C' + 4 * 'C' + 16 * 'G' - 1365] = 'P';

    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'U' - 1365] = 'H';
    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'C' - 1365] = 'H';
    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'A' - 1365] = 'Q';
    h_codonMap[1 * 'C' + 4 * 'A' + 16 * 'G' - 1365] = 'Q';

    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'U' - 1365] = 'R';
    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'C' - 1365] = 'R';
    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'A' - 1365] = 'R';
    h_codonMap[1 * 'C' + 4 * 'G' + 16 * 'G' - 1365] = 'R';

    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'U' - 1365] = 'I';
    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'C' - 1365] = 'I';
    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'A' - 1365] = 'I';
    h_codonMap[1 * 'A' + 4 * 'U' + 16 * 'G' - 1365] = 'M';

    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'U' - 1365] = 'T';
    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'C' - 1365] = 'T';
    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'A' - 1365] = 'T';
    h_codonMap[1 * 'A' + 4 * 'C' + 16 * 'G' - 1365] = 'T';

    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'U' - 1365] = 'N';
    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'C' - 1365] = 'N';
    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'A' - 1365] = 'K';
    h_codonMap[1 * 'A' + 4 * 'A' + 16 * 'G' - 1365] = 'K';

    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'U' - 1365] = 'S';
    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'C' - 1365] = 'S';
    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'A' - 1365] = 'R';
    h_codonMap[1 * 'A' + 4 * 'G' + 16 * 'G' - 1365] = 'R';

    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'U' - 1365] = 'V';
    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'C' - 1365] = 'V';
    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'A' - 1365] = 'V';
    h_codonMap[1 * 'G' + 4 * 'U' + 16 * 'G' - 1365] = 'V';

    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'U' - 1365] = 'A';
    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'C' - 1365] = 'A';
    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'A' - 1365] = 'A';
    h_codonMap[1 * 'G' + 4 * 'C' + 16 * 'G' - 1365] = 'A';

    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'U' - 1365] = 'D';
    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'C' - 1365] = 'D';
    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'A' - 1365] = 'E';
    h_codonMap[1 * 'G' + 4 * 'A' + 16 * 'G' - 1365] = 'E';

    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'U' - 1365] = 'G';
    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'C' - 1365] = 'G';
    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'A' - 1365] = 'G';
    h_codonMap[1 * 'G' + 4 * 'G' + 16 * 'G' - 1365] = 'G';

    cudaMemcpyToSymbol(codonMap, &h_codonMap, sizeof(char) * 421);

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
