#include "image_handler.h"
#include "seam_carving.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

imgProp_t* imgProp = (imgProp_t*)malloc(sizeof(imgProp_t));;
pixel_t* imgSrc;
energyPixel_t* imgGray;

int main(int argc, char** argv) {

	char* path = strcat(SOURCE_PATH, "castle_bmp.bmp");

	readBMP(imgSrc, imgGray, path, imgProp);
	map(imgGray, imgProp);

	return 0;
}