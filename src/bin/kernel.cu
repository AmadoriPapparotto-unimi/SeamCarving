#include "image_handler.h"
#include "seam_carving.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

void applySeamCarving(char *p) {

	pixel_t* imgSrc;
	imgProp_t* imgProp;
	energyPixel_t* imgGray;
	energyPixel_t* imgEnergy;
	seam_t* minSeam;

	FILE* f = fopen(p, "rb");
	if (f == NULL) {
		printf("*** FILE NOT FOUND ***\n");
		exit(1);
	}

	cudaMallocManaged(&imgProp, sizeof(imgProp_t));
	setupImgProp(imgProp, f);

	cudaMallocManaged(&imgSrc, imgProp->height * imgProp->width * sizeof(pixel_t));
	cudaMallocManaged(&imgGray, imgProp->height * imgProp->width * sizeof(energyPixel_t));
	cudaMallocManaged(&imgEnergy, imgProp->height * imgProp->width * sizeof(energyPixel_t));

	cudaMallocManaged(&minSeam, sizeof(seam_t));
	cudaMallocManaged(&minSeam->ids, imgProp->height * sizeof(int));

	readBMP(f, imgSrc, imgProp);
	//writeBMP_pixel(strcat(SOURCE_PATH, "hhh.bmp"), imgSrc, imgProp);
	toGrayScale(imgSrc, imgGray, imgProp);
	for (int i = 0; i < 1; i++) {
		map(imgGray, imgProp);
		printf("-----------------width %d height %d\n", imgProp->width, imgProp->height);
		findSeams(imgGray, imgProp, minSeam);
		removeSeam(imgGray, minSeam->ids, imgProp);
		printf("ITERAZIONE %d COMPLETATA\n", i);
	}

	setBMP_header(imgProp, 0, imgProp->width);

	pixel_t* img2convert = (pixel_t*)malloc(imgProp->imageSize * sizeof(pixel_t));
	energy2pixel(img2convert, imgGray, imgProp);
	writeBMP_pixel(strcat(SOURCE_PATH, "ffff.bmp"), img2convert, imgProp);
	free(img2convert);

	cudaFree(imgProp);
	cudaFree(imgGray);
	cudaFree(imgSrc);

	fclose(f);
}

int main(int argc, char** argv) {

	/*
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
	*/

	//imgProp_t* imgProp;

	char* path = strcat(SOURCE_PATH, "33.bmp");

	applySeamCarving(path);

	//cudaMallocManaged(imgGray, imgProp->imageSize);
	//map(imgGray, imgProp);
	cudaDeviceReset();
	return 0;
}

//void report_gpu_mem()
//{
//	size_t free, total;
//	cudaMemGetInfo(&free, &total);
//	printf("Free = %zu, Total = %zu\n", free, total);
//}
//
//int main()
//{
//	float* a, * a_out, *b, *bo;
//	int sz = 1 << 20; // 16Mb
//	report_gpu_mem();
//	cudaMallocManaged((void**)&a, sz);
//	report_gpu_mem();
//	cudaMallocManaged((void**)&a_out, sz);
//	report_gpu_mem();
//
//	cudaMallocManaged((void**)&b, sz);
//	report_gpu_mem();
//	cudaMallocManaged((void**)&bo, sz);
//	report_gpu_mem();
//
//
//	cudaFree(a);
//	report_gpu_mem();
//	cudaFree(a_out);
//	report_gpu_mem();
//	cudaFree(b);
//	report_gpu_mem();
//	cudaFree(bo);
//	report_gpu_mem();
//	return cudaDeviceReset();
//}