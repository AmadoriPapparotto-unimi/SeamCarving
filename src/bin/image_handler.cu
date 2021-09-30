#include "image_handler.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

ImgProp ip;

char grayValue(int r, int g, int b) {
	return (char) (r + g + b) / 3;
}

__global__ void toGrayScale(pel** img, pel** grayImg)
{
	printf("%d", threadIdx.x);
	//grayImg[threadIdx.x] = 
}

void setupImgProp(ImgProp* ip, FILE* f) {
	pel headerInfo[54];
	fread(headerInfo, sizeof(pel), 54, f);

	int width = *(int*)&headerInfo[18];
	int height = *(int*)&headerInfo[22];
	int rowBytes = (width * 3 + 3) & (~3);

	for (unsigned int i = 0; i < 54; i++)
		ip->headerInfo[i] = headerInfo[i];

	ip->height = height;
	ip->width = width;
	ip->rowBytes = rowBytes;
}

pel** ReadBMP(char* p) {
	//BMP LEGGE I PIXEL NEL FORMATO BGR
	FILE* f = fopen(p, "rb");
	if (f == NULL) {
		printf("*** FILE NOT FOUND ***\n");
		exit(1);
	}

	//extract information from headerInfo
	setupImgProp(&ip, f);
	printf("Input BMP dimension: (%u x %u)\n", ip.width, ip.height);

	pel **img, **imgGray;

	cudaMallocManaged(&img, ip.height * sizeof(pel*), 0);
	cudaMallocManaged(&imgGray, ip.height * sizeof(pel*), 0);

	for (unsigned int i = 0; i < ip.width; i++)
		cudaMallocManaged(&img[i], ip.rowBytes * sizeof(pel), 0);

	for (unsigned int i = 0; i < ip.width; i++)
		cudaMallocManaged(&imgGray[i], ip.width * sizeof(pel), 0);


	for (unsigned int i = 0; i < ip.height; i++) {
		fread(img[i], sizeof(pel), ip.rowBytes, f);
	}

	/*dim3 block;
	dim3 gg;
	block.x = 3;
	gg.x = 10;*/

	pel** 

	toGrayScale <<<block, gg >>> (img);

	fclose(f);
	return img;
}