#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "image_handler.h"
#include "seam_carving.h"

__global__
void min_(const seam_t* energiesArray, seam_t* outputArray, imgProp_t* imgProp, int nThreads);

__global__
void sum_(energyPixel_t* energyImg, seam_t* seam, int* out, imgProp_t* imgProp);

void report_gpu_mem();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %d\n", cudaGetErrorString(code), line);
        if (abort) exit(code);
    }
}
