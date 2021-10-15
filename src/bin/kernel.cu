#include "image_handler.h"
#include "seam_carving.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

void applySeamCarving(char *p) {

	pixel_t* imgSrc;
	imgProp_t* imgProp;
	energyPixel_t* imgGray;

	FILE* f = fopen(p, "rb");
	if (f == NULL) {
		printf("*** FILE NOT FOUND ***\n");
		exit(1);
	}

	cudaMallocManaged(&imgProp, sizeof(imgProp_t));
	setupImgProp(imgProp, f);

	cudaMallocManaged(&imgSrc, imgProp->height * imgProp->width * sizeof(pixel_t));
	cudaMallocManaged(&imgGray, imgProp->height * imgProp->width * sizeof(energyPixel_t));

	readBMP(f, imgSrc, imgProp);
	//writeBMP_pixel(strcat(SOURCE_PATH, "hhh.bmp"), imgSrc, imgProp);
	toGrayScale(imgSrc, imgGray, imgProp);
	map(imgGray, imgProp);
	findSeams(imgGray, imgProp);

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

	char* path = strcat(SOURCE_PATH, "a.bmp");

	applySeamCarving(path);

	//cudaMallocManaged(imgGray, imgProp->imageSize);
	//map(imgGray, imgProp);

	return 0;
}