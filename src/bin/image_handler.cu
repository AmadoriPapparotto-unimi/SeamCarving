#include "image_handler.h"
#include "seam_carving.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#define MAX_THREAD 1024


__device__ void grayValue(pixel_t *res, pel_t r, pel_t g, pel_t b) {
	int grayVal = (r + g + b) / 3;
	res->R = grayVal;
	res->G = grayVal;
	res->B = grayVal;
}

__global__ void toGrayScale(pixel_t* img, energyPixel_t* imgGray, int imageSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == gridDim.x * 1024 + 1)
		printf("%d", gridDim.x);

	if (id < imageSize) {
		grayValue(&imgGray[id].pixel, img[id].R, img[id].G, img[id].B);
	}
}

void setupImgProp(imgProp_t* ip, FILE* f) {
	pel_t headerInfo[54];
	fread(headerInfo, sizeof(pel_t), 54, f);

	int width = *(int*)&headerInfo[18];
	int height = *(int*)&headerInfo[22];
	printf("#bytes: %d\n", *(int*)&headerInfo[34]);

	for (unsigned int i = 0; i < 54; i++)
		ip->headerInfo[i] = headerInfo[i];

	ip->height = height;
	ip->width = width;
	ip->imageSize = width * height;
}

void readBMP(pixel_t* img, energyPixel_t* imgGray, char* p, imgProp_t* ip) {

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
	setupImgProp(ip, f);
	printf("Input BMP dimension: (%u x %u)\n", ip->width, ip->height);

	cudaMallocManaged(&img, ip->height * ip->width * sizeof(pixel_t));
	cudaMallocManaged(&imgGray, ip->height * ip->width * sizeof(energyPixel_t));

	for (unsigned int i = 0; i < ip->height * ip->width; i++) {
		fread(&img[i], sizeof(pel_t), sizeof(pixel_t), f);
	}

	dim3 blocks;
	blocks.x = ip->imageSize / MAX_THREAD + 1;

	toGrayScale << <blocks, MAX_THREAD >> > (img, imgGray, ip->imageSize);
	cudaDeviceSynchronize();
	writeBMP_pixel(strcat(SOURCE_PATH, "gray.bmp"), energy2pixel(imgGray, ip), ip);

	fclose(f);

	map(imgGray, ip);
	findSeams(imgGray, ip);

}

void writeBMP_pixel(char* p, pixel_t* img, imgProp_t* ip) {
	FILE* fw = fopen(p, "wb");

	fwrite(ip->headerInfo, sizeof(pel_t), 54, fw);
	fwrite(img, sizeof(pixel_t), ip->imageSize, fw);

	fclose(fw);
	printf("Immagine %s generata\n", p);
}

void writeBMP_energy(char* p, energyPixel_t* energyImg, imgProp_t* ip) {
	pixel_t* img;
	int sd = 1;
	img = (pixel_t*)malloc(ip->imageSize * sizeof(pixel_t));

	for (int i = 0; i < ip->imageSize; i++) {
		img[i].R = energyImg[i].energy;
		img[i].G = energyImg[i].energy;
		img[i].B = energyImg[i].energy;
	}

	writeBMP_pixel(p, img, ip);
}

void writeBMP_minimumSeam(char* p, energyPixel_t* energyImg, seam_t* minSeam, imgProp_t* imgProp) {
	for (int y = 0; y < imgProp->height; y++) {
		printf("PATH: %d\n", minSeam[0].ids[y]);
		energyImg[minSeam[0].ids[y]].pixel.R = 0;
		energyImg[minSeam[0].ids[y]].pixel.G = 255;
		energyImg[minSeam[0].ids[y]].pixel.B = 0;
	}
	writeBMP_pixel(strcat(SOURCE_PATH, "seams_map_minimum.bmp"), energy2pixel(energyImg, imgProp), imgProp);
}

pixel_t* energy2pixel(energyPixel_t* energyImg, imgProp_t* ip) {
	pixel_t* img;
	img = (pixel_t*)malloc(ip->imageSize * sizeof(pixel_t));

	for (int i = 0; i < ip->imageSize; i++) {
		img[i] = energyImg[i].pixel;
	}

	return img;
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