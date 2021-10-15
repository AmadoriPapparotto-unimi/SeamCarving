#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "image_handler.h"
#include "utils.h"
#include "seam_carving.h"

//static const int blockSize = 1024;

__global__ void min_(const seam_t* energiesArray, seam_t* outputArray, imgProp_t* imgProp) {
    int thIdx = threadIdx.x;
    const int myBlockSize = 1024;
    int gthIdx = thIdx + blockIdx.x * myBlockSize;
    __shared__ seam_t shArr[myBlockSize];
    if(gthIdx < imgProp->width)
        shArr[thIdx] = energiesArray[gthIdx];

    int seamsPerBlock = myBlockSize; // 0 < seamsPerBlock < 1024
    
    // si ottiene il numero preciso di seams rimanenti da controllare:
    // per ogni blocco che non sia l'ultimo -> seamsPerBlock = 1024
    // per ultimo blocco -> seamsPerBlock = differenza imgProp->width - (1024 * numBlocchi - 1)
    if (1024 * (blockIdx.x + 1) > imgProp->width)
        seamsPerBlock = imgProp->width - 1024 * blockIdx.x;

    __syncthreads();
    
    int size = seamsPerBlock / 2;
    bool isOdd = seamsPerBlock % 2 == 1;
    if (isOdd) {
        size++;
        if (thIdx < seamsPerBlock / 2)
            shArr[thIdx] = (shArr[thIdx].total_energy < shArr[thIdx + size].total_energy) ? shArr[thIdx] : shArr[thIdx + size];

        size /= 2;
    }
    // get minimum
    for (; size > 0; size /= 2) { //uniform
        if (thIdx < size)
            shArr[thIdx] = (shArr[thIdx].total_energy < shArr[thIdx + size].total_energy) ? shArr[thIdx] : shArr[thIdx + size];
        __syncthreads();
    }

    //save current block's minimum
    if (thIdx == 0) {
        outputArray[blockIdx.x] = shArr[0];
    }
}

void minArr(dim3 gridSize, dim3 blockSize, seam_t* energiesArray, seam_t* outputArray, imgProp_t* imgProp) {
    min_ << <gridSize, blockSize >> > (energiesArray, outputArray, imgProp);
}

void report_gpu_mem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Free = %zu, Total = %zu\n", free, total);
}
