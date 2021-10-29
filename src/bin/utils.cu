#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "image_handler.h"
#include "utils.cuh"
#include "seam_carving.h"

__global__ 
void min_(const seam_t* energiesArray, seam_t* outputArray, imgProp_t* imgProp, int nThreads) {

    /// <summary>
    /// Kernel GPU che permette di calcolare il seam minimo tra tutti quelli trovati.
    /// Sfrutta la shared memory e la parallel reduction, al fine di massimizzare le performance
    /// </summary>
    /// <param name="energiesArray"></param>
    /// <param name="outputArray"></param>
    /// <param name="imgProp"></param>
    /// <param name="nThreads"></param>
    /// 
    /// <returns></returns>
    int thIdx = threadIdx.x;
    const int myBlockSize = nThreads;
    int gthIdx = thIdx + blockIdx.x * myBlockSize;

    extern __shared__ int shArrMin[];
    int* shared_mins = (int*)shArrMin;
    int* shared_min_indices = (int*)(&(shArrMin[nThreads]));

    if (gthIdx < imgProp->width) {
        shared_mins[thIdx] = energiesArray[gthIdx].total_energy;
        shared_min_indices[thIdx] = gthIdx;
    }
    

    int seamsPerBlock = myBlockSize; // 0 < seamsPerBlock < 1024
    
    // si ottiene il numero preciso di seams rimanenti da controllare:z
    // per ogni blocco che non sia l'ultimo -> seamsPerBlock = 1024
    // per ultimo blocco -> seamsPerBlock = differenza imgProp->width - (1024 * numBlocchi - 1)
    if (myBlockSize * (blockIdx.x + 1) > imgProp->width)
        seamsPerBlock = imgProp->width - myBlockSize * blockIdx.x;

    __syncthreads();
    
    int size = seamsPerBlock;   //ogni volta si dimezza la grandezza dell'array finale
    bool isOdd;
    //if (isOdd) {
    //    size++;
    //    if (thIdx < seamsPerBlock / 2) {
    //        if (shared_mins[thIdx] > shared_mins[thIdx + size]) {
    //            shared_mins[thIdx] = shared_mins[thIdx + size];
    //            shared_min_indices[thIdx] = shared_min_indices[thIdx + size];
    //        }
    //    }
    //    size /= 2;
    //}
    // get minimum
    while (size > 0) { //uniform
        
        isOdd = size % 2 == 1 && size != 1; //bisogna distinguere se il numero di seam di cui trovare il minimo è pari o dispari, questo perchè un elemento ne rimarrebbe escluso
        size /= 2;
        if (isOdd) {
            size++;
        }

        if (thIdx < size) {
            if (isOdd && thIdx == size - 1) {
                continue;
            }
            if (shared_mins[thIdx] > shared_mins[thIdx + size]) {
                shared_mins[thIdx] = shared_mins[thIdx + size];
                shared_min_indices[thIdx] = shared_min_indices[thIdx + size];
            }
        }
        __syncthreads();

    }

    //while (size > 32) ...
    // ...
    //if (thIdx < 32) {
    //    volatile int* vmem = shared_min_indices;
    //    vmem[thIdx] = shared_mins[thIdx] < shared_mins[thIdx + 32] ? shared_min_indices[thIdx] : shared_min_indices[thIdx + 32];
    //    vmem[thIdx] = shared_mins[thIdx] < shared_mins[thIdx + 16] ? shared_min_indices[thIdx] : shared_min_indices[thIdx + 16];
    //    vmem[thIdx] = shared_mins[thIdx] < shared_mins[thIdx + 8] ? shared_min_indices[thIdx] : shared_min_indices[thIdx + 8];
    //    vmem[thIdx] = shared_mins[thIdx] < shared_mins[thIdx + 4] ? shared_min_indices[thIdx] : shared_min_indices[thIdx + 4];
    //    vmem[thIdx] = shared_mins[thIdx] < shared_mins[thIdx + 2] ? shared_min_indices[thIdx] : shared_min_indices[thIdx + 2];
    //    vmem[thIdx] = shared_mins[thIdx] < shared_mins[thIdx + 1] ? shared_min_indices[thIdx] : shared_min_indices[thIdx + 1];
    //}

    //save current block's minimum
    if (thIdx == 0) {
        outputArray[blockIdx.x] = energiesArray[shared_min_indices[0]];
        //printf("seams %f\n", outputArray[blockIdx.x].total_energy);

    }
}

__global__
void sum_(energyPixel_t* energyImg, seam_t* seam, int* out, imgProp_t* imgProp) {

    int thIdx = threadIdx.x;
    const int myBlockSize = blockDim.x;
    int gthIdx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int shArrSum[];
    int* shared_sum = (int*)shArrSum;

    if (gthIdx < imgProp->height) {
        shared_sum[thIdx] = energyImg[seam->ids[gthIdx]].energy;
        //printf("t: %d, %d\n", gthIdx, shared_sum[thIdx]);
    }


    int seamsPerBlock = myBlockSize; // 0 < seamsPerBlock < 1024

    // si ottiene il numero preciso di seams rimanenti da controllare:z
    // per ogni blocco che non sia l'ultimo -> seamsPerBlock = 1024
    // per ultimo blocco -> seamsPerBlock = differenza imgProp->width - (1024 * numBlocchi - 1)
    if (myBlockSize * (blockIdx.x + 1) > imgProp->height)
        seamsPerBlock = imgProp->height - myBlockSize * blockIdx.x;

    __syncthreads();

    int size = seamsPerBlock;   //ogni volta si dimezza la grandezza dell'array finale
    bool isOdd;

    while (size > 0) { //uniform
        isOdd = size % 2 == 1 && size != 1; //bisogna distinguere se il numero di seam di cui trovare il minimo è pari o dispari, questo perchè un elemento ne rimarrebbe escluso
        size /= 2;
        if (isOdd) {
            size++;
        }

        if (thIdx < size) {
            if (isOdd && thIdx == size - 1) {
                continue;
            }
            shared_sum[thIdx] += shared_sum[thIdx + size];

        }
        __syncthreads();

    }

    //save current block's minimum
    if (thIdx == 0) {
        out[blockIdx.x] = shared_sum[0];
    }
}

void report_gpu_mem()
{

    /// <summary>
    /// Funzione di supporto che permette di stampare la memoria GPU libera e occupata 
    /// </summary>
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Free = %zu, Total = %zu\n", free, total);
}
