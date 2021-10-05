#include "image_handler.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#define MAX_THREAD 1024

imgProp ip;

__device__ void grayValue(pixel *res, pel r, pel g, pel b) {
	int grayVal = (r + g + b) / 3;
	res->R = grayVal;
	res->G = grayVal;
	res->B = grayVal;
}

__global__ void toGrayScale(pixel* img, pixel* imgGray, int imageSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imageSize) {
		grayValue(&imgGray[id], img[id].R, img[id].G, img[id].B);
	}
}

void setupImgProp(imgProp* ip, FILE* f) {
	pel headerInfo[54];
	fread(headerInfo, sizeof(pel), 54, f);

	int width = *(int*)&headerInfo[18];
	int height = *(int*)&headerInfo[22];
	printf("%d\n", *(int*)&headerInfo[34]);

	for (unsigned int i = 0; i < 54; i++)
		ip->headerInfo[i] = headerInfo[i];

	ip->height = height;
	ip->width = width;
	ip->imageSize = width * height;
}

pixel* readBMP(char* p) {

	//img[0] = B
	//img[1] = G
	//img[2] = R
	//BMP LEGGE I PIXEL NEL FORMATO BGR
	FILE* f = fopen(p, "rb");
	if (f == NULL) {
		printf("*** FILE NOT FOUND ***\n");
		exit(1);
	}

	//extract information from headerInfo
	setupImgProp(&ip, f);
	printf("Input BMP dimension: (%u x %u)\n", ip.imageSize, ip.height);

	pixel* img;
	pixel* imgGray;

	cudaMallocManaged(&img, ip.height * ip.width * sizeof(pixel));
	cudaMallocManaged(&imgGray, ip.height * ip.width * sizeof(pixel));

	for (unsigned int i = 0; i < ip.height * ip.width; i++) {
		fread(&img[i], sizeof(pel), sizeof(pixel), f);
	}

	dim3 blocks;
	blocks.x = ip.imageSize / MAX_THREAD;

	toGrayScale << <blocks, MAX_THREAD >> > (img, imgGray, ip.imageSize);
	cudaDeviceSynchronize();
	writeBMP_pixel(strcat(SOURCE_PATH, "created.bmp"), ip, imgGray);

	fclose(f);
	return img;
}

void writeBMP_pixel(char* p, imgProp imgProp, pixel* img) {
	FILE* fw = fopen(p, "wb");

	fwrite(imgProp.headerInfo, sizeof(pel), 54, fw);
	fwrite(img, sizeof(pixel), imgProp.imageSize, fw);

	fclose(fw);
}

//void writeBMP_pel(char* p, imgProp imgProp, pel* img) {
//	FILE* fw = fopen(p, "wb");
//
//	//0000 0000 0001 0101 0001 0111 1010 0000
//	imgProp.headerInfo[34] = ip.imageSize >> 0;//0xa0;
//	imgProp.headerInfo[35] = ip.imageSize >> 8;//0x17;
//	imgProp.headerInfo[36] = ip.imageSize >> 16;//0x15;
//	imgProp.headerInfo[37] = ip.imageSize >> 24;//0x0;
//	
//	printf("%ld; %d", imgProp.headerInfo[34], imgProp.height * imgProp.width);
//	fwrite(imgProp.headerInfo, sizeof(pel), 54, fw);
//	fwrite(img, sizeof(pel), imgProp.height * imgProp.width, fw);
//
//	fclose(fw);
//}