#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "image_handler.h"

typedef struct seamStruct {
	int total_energy;
	int* ids;
} seam_t;

void minArr(dim3 gridSize, dim3 blockSize, seam_t* seams, seam_t* outputSeams, imgProp_t* imgProp);