#include "image_handler.h"
#include "seam_carving.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

pixel_t* imgSrc;
energyPixel_t* imgGray;

int main(int argc, char** argv) {

	imgProp_t* imgProp;
	cudaMallocManaged(&imgProp, sizeof(imgProp_t));

	char* path = strcat(SOURCE_PATH, "castle_bmp.bmp");

	readBMP(imgSrc, imgGray, path, imgProp);
	//cudaMallocManaged(imgGray, imgProp->imageSize);
	//map(imgGray, imgProp);

	return 0;
}