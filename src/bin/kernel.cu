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

	gpuErrchk(cudaMallocManaged(&imgProp, sizeof(imgProp_t)));
	setupImgProp(imgProp, f);

	gpuErrchk(cudaMallocManaged(&imgSrc, imgProp->height * imgProp->width * sizeof(pixel_t)));
	gpuErrchk(cudaMallocManaged(&imgGray, imgProp->height * imgProp->width * sizeof(energyPixel_t)));
	gpuErrchk(cudaMallocManaged(&imgEnergy, imgProp->height * imgProp->width * sizeof(energyPixel_t)));

	gpuErrchk(cudaMallocManaged(&minSeam, sizeof(seam_t)));
	gpuErrchk(cudaMallocManaged(&minSeam->ids, imgProp->height * sizeof(int)));

	readBMP(f, imgSrc, imgProp);
	//writeBMP_pixel(strcat(SOURCE_PATH, "hhh.bmp"), imgSrc, imgProp);
	toGrayScale(imgSrc, imgGray, imgProp);
	
	for (int i = 0; i < 15; i++) {
		map(imgGray, imgProp);
		printf("-----------------width %d height %d\n", imgProp->width, imgProp->height);
		findSeams(imgGray, imgProp, minSeam);
		
		for (int i = 0; i < imgProp->height; i++) {
			printf("%d - ", minSeam[0].ids[i]);
		}
		removeSeam(imgGray, minSeam, imgProp);
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

	char* path = strcat(SOURCE_PATH, "castle_bmp.bmp");

	applySeamCarving(path);

	//cudaMallocManaged(imgGray, imgProp->imageSize);
	//map(imgGray, imgProp);
	cudaDeviceReset();
	return 0;
}
