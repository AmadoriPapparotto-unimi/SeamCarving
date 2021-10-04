#include "image_handler.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

imgProp ip;

pel grayValue(pel r, pel g, pel b) {
	return (pel) (r + g + b) / 3;
}

__global__ void toGrayScale(pel* img)
{
	int id = threadIdx.x; //TODO: LINEARIZZARE INDICE
	//printf("%d - %d - %d\n", id, img[1][0], *(pel*)img[1428]);
	//*(pel*)grayImg[id] = grayValue(*(pel*)grayImg[id + 2], *(pel*)grayImg[id + 1], *(pel*)grayImg[1]);
}

void setupImgProp(imgProp* ip, FILE* f) {
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
	printf("Input BMP dimension: (%u x %u): %u\n", ip.width, ip.height, ip.rowBytes);

	pixel* img;

	cudaMallocManaged(&img, ip.height * ip.width * sizeof(pixel));

	for (unsigned int i = 0; i < ip.height * ip.width; i++) {
		fread(&img[i], sizeof(pel), sizeof(pixel), f);
	}

	writeBMP("src/assets/images/created.bmp", ip, img);

	fclose(f);
	return img;
}

void writeBMP(char* p, imgProp imgProp, pixel* img) {
	FILE* fw = fopen(p, "wb");

	fwrite(imgProp.headerInfo, sizeof(pel), 54, fw);
	fwrite(img, sizeof(pixel), imgProp.height * imgProp.width, fw);
	fclose(fw);
}