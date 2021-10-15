#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "image_handler.h"
#include "seam_carving.h"

void minArr(dim3 gridSize, dim3 blockSize, seam_t* seams, seam_t* outputSeams, imgProp_t* imgProp);
void report_gpu_mem();