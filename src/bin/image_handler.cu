#include "image_handler.h"
#include "seam_carving.h"
#include "utils.cuh"
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

__device__ 
void grayValue(energyPixel_t *energyPixel, pel_t r, pel_t g, pel_t b, int id) {
	int grayVal = (r + g + b) / 3;
	energyPixel->color = grayVal;
	energyPixel->idPixel = id;
}

__global__ 
void toGrayScale_(pixel_t* img, energyPixel_t* imgGray, int imageSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < imageSize) {
		grayValue(&imgGray[id], img[id].R, img[id].G, img[id].B, id);
	}
}

__global__
void generateEnergyImg_(pixel_t* imgSrc, energyPixel_t* energyImg, imgProp_t* imgProp) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgSrc[idThread].R = energyImg[idThread].energy;
		imgSrc[idThread].G = energyImg[idThread].energy;
		imgSrc[idThread].B = energyImg[idThread].energy;
	}
}

__global__
void energy2pixel_(pixel_t* imgSrc, energyPixel_t* energyImg, imgProp_t* imgProp) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgSrc[idThread].R = energyImg[idThread].color;
		imgSrc[idThread].G = energyImg[idThread].color;
		imgSrc[idThread].B = energyImg[idThread].color;
	}
}

__global__
void colorSeamToRemove_(pixel_t* img, seam_t* seam, imgProp_t* imgProp) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		img[seam->ids[idThread]].R = 0;
		img[seam->ids[idThread]].G = 255;
		img[seam->ids[idThread]].B = 0;
	}
}

void toGrayScale(pixel_t* img, energyPixel_t* imgGray, imgProp_t* imgProp) {
	int blocks = imgProp->imageSize / 1024 + 1;

	toGrayScale_<< <blocks, 1024 >> > (img, imgGray, imgProp->imageSize);
	gpuErrchk(cudaDeviceSynchronize());
}

void setupImgProp(imgProp_t* imgProp, FILE* f) {
	pel_t headerInfo[54];
	fread(headerInfo, sizeof(pel_t), 54, f);


	int width = *(int*)&headerInfo[18];
	int height = *(int*)&headerInfo[22];
	printf("#bytes: %d\n", *(int*)&headerInfo[34]);

	for (unsigned int i = 0; i < 54; i++)
		imgProp->headerInfo[i] = headerInfo[i];

	imgProp->height = height;
	imgProp->width = width;
	imgProp->imageSize = width * height;

	printf("Input BMP dimension: (%u x %u)\n", imgProp->width, imgProp->height);
	printf("IHeader[2] %d\n", *(int*)&headerInfo[2]);
}

void readBMP(FILE *f, pixel_t* img, imgProp_t* imgProp) {
	//img[0] = B
	//img[1] = G
	//img[2] = R
	//BMP LEGGE I PIXEL NEL FORMATO BGR

	for (int r = 0; r < imgProp->height; r++) {
		fread(&img[r*imgProp->width], sizeof(pel_t), imgProp->width * sizeof(pixel_t), f);

		int padding = 4 - ((imgProp->width * 3) % 4);
		if (padding != 0 && padding != 4) {
			fseek(f, padding, SEEK_CUR);
			
		}
	}

}

void writeBMP_pixel(char* p, pixel_t* img, imgProp_t* ip) {
	FILE* fw = fopen(p, "wb");

	printf("FINAL HEIGHT %d\n", ip->height);
	printf("FINAL WIDTH %d\n", ip->width);
	fwrite(ip->headerInfo, 1, 54, fw);

	int padding = 0;
	for (int r = 0; r < ip->height; r++) {
		for (int c = 0; c < ip->width; c++) {
			fputc(img[c + r * ip->width].B, fw);
			fputc(img[c + r * ip->width].G, fw);
			fputc(img[c + r * ip->width].R, fw);
		}
		padding = 4 - ((ip->width * 3) % 4);
		if (padding != 0 && padding != 4) {
			for (int i = 0; i < padding; i++) {
				fputc(0, fw);
			}
		}
	}
	fflush(fw);

	fclose(fw);
	printf("Immagine %s generata\n", p);
}

void writeBMP_energy(char* p, energyPixel_t* energyImg, imgProp_t* imgProp) {
	pixel_t* img;
	cudaMallocManaged(&img, imgProp->imageSize * sizeof(pixel_t));

	generateEnergyImg_ << <imgProp->imageSize/1024 + 1, 1024>> > (img, energyImg, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

	writeBMP_pixel(p, img, imgProp);

	gpuErrchk(cudaFree(img));
}

void writeBMP_grayscale(energyPixel_t* imgGray, imgProp_t* imgProp) {
	pixel_t* img2convert;

	cudaMallocManaged(&img2convert, imgProp->imageSize * sizeof(pixel_t));

	energy2pixel_ << <imgProp->imageSize/1024 + 1, 1024 >> > (img2convert, imgGray, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

	writeBMP_pixel("C:/aa/gray.bmp", img2convert, imgProp);
	gpuErrchk(cudaFree(img2convert));
}

void writeBMP_minimumSeam(char* p, pixel_t* img, seam_t* minSeam, imgProp_t* imgProp) {

	colorSeamToRemove_ << <imgProp->height/1024 + 1, 1024>> > (img, minSeam, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

	writeBMP_pixel("C:/aa/seams_map_minimum.bmp", img, imgProp);
	gpuErrchk(cudaFree(img));
}

void setBMP_header(imgProp_t* imgProp, int fileSize, int width) {
	imgProp->headerInfo[2] = (unsigned char)(fileSize >> 0) & 0xff;
	imgProp->headerInfo[3] = (unsigned char)(fileSize >> 8) & 0xff;
	imgProp->headerInfo[4] = (unsigned char)(fileSize >> 16) & 0xff;
	imgProp->headerInfo[5] = (unsigned char)(fileSize >> 24) & 0xff;
	
	imgProp->headerInfo[18] = (unsigned char)(width >> 0) & 0xff;
	imgProp->headerInfo[19] = (unsigned char)(width >> 8) & 0xff;
	imgProp->headerInfo[20] = (unsigned char)(width >> 16) & 0xff;
	imgProp->headerInfo[21] = (unsigned char)(width >> 24) & 0xff;
}