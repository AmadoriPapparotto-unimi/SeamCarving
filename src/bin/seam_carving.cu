#include "image_handler.h"
#include "seam_carving.h"
#include "math.h"
#include "utils.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

__device__ 
char getPosition(int id, imgProp_t* imgProp) {
	int width = imgProp->width;
	int imageSize = imgProp->imageSize;

	//ANGOLO BASSO SX:									[0]
	//ANGOLO ALTO SX									[1]
	//COLONNA SX: id % imgProp->width == 0				[2]
	//ANGOLO BASSO DX									[3]
	//ANGOLO ALTO DX									[4]
	//COLONNA DX: id + 1 %  imgProp->width == 0			[5]
	//RIGA DOWN: id < imgProp->width					[6]
	//RIGA UP: id > imgProp.imageSize - imgProp->width	[7]
	//IN MEZZO											[-1]
	/*
		08 09 10 11
		04 05 06 07
		00 01 02 03
	*/

	if (id % width == 0) {
		if (id == 0)
			return 0;
		if (id == imageSize - width)
			return 1;
		return 2;
	}
	else if (id % width == width - 1) {
		if (id == width - 1)
			return 3;
		if (id == imageSize - 1)
			return 4;
		return 5;
	}
	else if (id < width)
		return 6;
	else if (id > imageSize - width)
		return 7;
	return -1;
}

__device__ 
void calculateEnergy(energyPixel_t* energyPixel, energyPixel_t* pixel, int id, imgProp_t* imgProp) {
	int dx2, dy2;
	//ANGOLO BASSO SX:									[0]
	//ANGOLO ALTO SX									[1]
	//COLONNA SX: id % imgProp->width == 0				[2]
	//ANGOLO BASSO DX									[3]
	//ANGOLO ALTO DX									[4]
	//COLONNA DX: id + 1 %  imgProp->width == 0			[5]
	//RIGA DOWN: id < imgProp->width					[6]
	//RIGA UP: id > imgProp.imageSize - imgProp->width	[7]
	//IN MEZZO											[-1]
	/*
		678
		345
		012
	*/

	char pos = getPosition(id, imgProp);
	switch (pos)
	{
	case 0:
		dx2 = energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color;
		break;
	case 1:
		dx2 = energyPixel[id + 1].color;
		dy2 = energyPixel[id - imgProp->width].color;
		break;
	case 2:
		dx2 = energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color - energyPixel[id - imgProp->width].color;
		break;
	case 3:
		dx2 = energyPixel[id - 1].color;
		dy2 = energyPixel[id + imgProp->width].color;
		break;
	case 4:
		dx2 = energyPixel[id - 1].color;
		dy2 = energyPixel[id - imgProp->width].color;
		break;
	case 5:
		dx2 = energyPixel[id - 1].color;
		dy2 = energyPixel[id + imgProp->width].color - energyPixel[id - imgProp->width].color;
		break;
	case 6:
		dx2 = energyPixel[id - 1].color - energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color;
		break;
	case 7:
		dx2 = energyPixel[id - 1].color - energyPixel[id + 1].color;
		dy2 = energyPixel[id - imgProp->width].color;
		break;
	case -1:
		dx2 = energyPixel[id - 1].color - energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color - energyPixel[id - imgProp->width].color;
		break;
	}

	pixel->energy = sqrtf((dx2 * dx2) + (dy2 * dy2));
}

__device__ 
int min(int id1, int id2, energyPixel_t* energyImg)
{
	return (energyImg[id1].energy < energyImg[id2].energy) ? id1 : id2;
}


__global__
void energyMap_(energyPixel_t* energyImg, imgProp_t* imgProp) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imgProp->imageSize) {
		calculateEnergy(energyImg, &energyImg[id], id, imgProp);
	}
}

__global__ 
void computeSeams_(energyPixel_t* energyImg, pixel_t* imgSrc, seam_t* seams, imgProp_t* imgProp, bool colorSeams = false) {

	//ANGOLO BASSO SX:									[0]
	//ANGOLO ALTO SX									[1]
	//COLONNA SX: id % imgProp->width == 0				[2]
	//ANGOLO BASSO DX									[3]
	//ANGOLO ALTO DX									[4]
	//COLONNA DX: id + 1 %  imgProp->width == 0			[5]
	//RIGA DOWN: id < imgProp->width					[6]
	//RIGA UP: id > imgProp.imageSize - imgProp->width	[7]
	//IN MEZZO											[-1]
	/*
		678
		345
		012
	*/

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;

	int currentId = idThread;
	if (currentId > imgProp->width - 1)
		return;
	int nextIdMin = currentId;


	seams[idThread].total_energy = 0;
	//seams[idThread].ids[0] = currentId;



	for (int i = 0; i < imgProp->height; i++) {		
		
		currentId = nextIdMin;

		seams[idThread].total_energy += energyImg[currentId].energy;
		seams[idThread].ids[i] = currentId;
		if (colorSeams) {
			imgSrc[currentId].R = 255;
			imgSrc[currentId].B = 0;
			imgSrc[currentId].G = 0;
		}

		int pos = getPosition(currentId, imgProp);
		switch (pos)
		{
		case 0:
		case 2:
			nextIdMin = min(currentId + imgProp->width, currentId + 1 + imgProp->width, energyImg);
			break;
		case 3:
		case 5:
			nextIdMin = min(currentId + imgProp->width, currentId - 1 + imgProp->width, energyImg);
			break;
		case 1:
		case 7:
			break;
		default:
			nextIdMin =  min(min(currentId + imgProp->width, currentId - 1 + imgProp->width, energyImg),
				currentId + 1 + imgProp->width, energyImg);
			break;
		}

	}
}

__global__
void populatePixelsToRemove_(int* pixelsToRemove, int* currentMinSeamIds, int totalIterations, int currentIteration) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;

	pixelsToRemove[idThread * totalIterations + currentIteration] = currentMinSeamIds[idThread];
}

__global__
void removeSeam_(energyPixel_t* energyImg, int* idsToRemove, imgProp_t* imgProp, energyPixel_t* newImageGray) {

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		int idRow = idThread / imgProp->width;
		int idToRemove = idsToRemove[idRow];
		int shift = idThread < idToRemove ? idRow : idRow + 1;

		if (idThread == idToRemove)
			return;

		int newPosition = idThread - shift;

		newImageGray[newPosition].energy = energyImg[idThread].energy;
		newImageGray[newPosition].color = energyImg[idThread].color;
		newImageGray[newPosition].idPixel = energyImg[idThread].idPixel;
	}
}

__global__
void updateImageGray_(energyPixel_t* imgGray, energyPixel_t* imgWithoutSeamGray, imgProp_t* imgProp) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgGray[idThread].color = imgWithoutSeamGray[idThread].color;
		imgGray[idThread].idPixel = imgWithoutSeamGray[idThread].idPixel;
	}
}

__global__
void updateImageColored_(pixel_t* imgSrc, pixel_t* imgWithoutSeamSrc, imgProp_t* imgProp) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgSrc[idThread].R = imgWithoutSeamSrc[idThread].R;
		imgSrc[idThread].G = imgWithoutSeamSrc[idThread].G;
		imgSrc[idThread].B = imgWithoutSeamSrc[idThread].B;
	}
}

__global__
void removePixelsFromSrc_(pixel_t* imgSrc, pixel_t* newImgSrc, energyPixel_t* imgGray, imgProp_t* imgProp) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		newImgSrc[idThread].R = imgSrc[imgGray[idThread].idPixel].R;
		newImgSrc[idThread].G = imgSrc[imgGray[idThread].idPixel].G;
		newImgSrc[idThread].B = imgSrc[imgGray[idThread].idPixel].B;
	}
}


void energyMap(energyPixel_t* energyImg, imgProp_t* imgProp) {
	energyMap_ << <imgProp->imageSize / 1024 + 1, 1024 >> > (energyImg, imgProp);
	gpuErrchk(cudaDeviceSynchronize());
	//writeBMP_energy("src/assets/images/energy.bmp", energyImg, imgProp);
}

void findSeams(energyPixel_t* energyImg, pixel_t* imgSrc, imgProp_t* imgProp, seam_t *minSeam, seam_t* seams, seam_t* minSeamsPerBlock) {
	int nThreads = 128;
	//energyPixel_t* img;
	int numBlocksComputeSeams = imgProp->width / nThreads + 1;
	int numBlocksMin = imgProp->width / 1024 + 1;


	//gpuErrchk(cudaMallocManaged(&img, imgProp->imageSize * sizeof(energyPixel_t)));
	//for (int i = 0; i < imgProp->imageSize; i++) {
	//	img[i].pixel.R = energyImg[i].energy;
	//	img[i].pixel.G = energyImg[i].energy;
	//	img[i].pixel.B = energyImg[i].energy;
	//	img[i].energy = energyImg[i].energy;
	//}


	computeSeams_ << <numBlocksComputeSeams, nThreads >> > (energyImg, imgSrc, seams, imgProp);
	//pixel_t* img2convert = (pixel_t*)malloc(imgProp->imageSize * sizeof(pixel_t));
	//energy2pixel(img2convert, img, imgProp);
	//writeBMP_pixel(strcat(SOURCE_PATH,"seams_map.bmp"), img2convert, imgProp);

	min_ <<<numBlocksMin, 1024, 1024 * (sizeof(int) + sizeof(int))>>>(seams, minSeamsPerBlock, imgProp, 1024);
	gpuErrchk(cudaDeviceSynchronize());

	*minSeam = minSeamsPerBlock[0];
	for (int i = 1; i < numBlocksMin; i++) {
		if (minSeamsPerBlock[i].total_energy < minSeam->total_energy) {
			*minSeam = minSeamsPerBlock[i];
		}
	}

	//printf("min: %d\n", minSeam->ids[0]);

	//populatePixelsToRemove_ << < imgProp->height/nThreads + 1, nThreads>> > ();
	
	//minArr(1, imgProp->width / 1024 + 1, minSeamsPerBlock, minSeam, imgProp, imgProp->width / 1024 + 1);
	//gpuErrchk(cudaDeviceSynchronize());


	//dummyMin(seams, *minSeam, imgProp);

	//printf("%d - \n", minSeam[0].total_energy);
	

	//for (int i = 0; i < imgProp->height; i++) {
	//	printf("%d - ", minSeam->ids[i]);
	//}
	//for (int y = 0; y < imgProp->height; y++) {
	//	img[minSeam[0].ids[y]].pixel.R = 0;
	//	img[minSeam[0].ids[y]].pixel.G = 255;
	//	img[minSeam[0].ids[y]].pixel.B = 0;
	//}

	//energy2pixel(img2convert, img, imgProp);
	//writeBMP_pixel(strcat(SOURCE_PATH, "seams_map_minimum.bmp"), img2convert, imgProp);
	//free(img2convert);

	//printf("%d", minSeamPath[0].total_energy);
	//for (int i = 0; i < imgProp->width; i++)
	//gpuErrchk(cudaFree(&(seams[i].ids)));

	//for (int i = 0; i < numBlocks; i++)
	//gpuErrchk(cudaFree(&minSeamsPerBlock[i].ids));

	//gpuErrchk(cudaFree(img));
}

void removeSeam(energyPixel_t* imgGray, energyPixel_t* imgWithoutSeamGray, seam_t* idsToRemove, imgProp_t* imgProp) {

	int newImgSizePixel = imgProp->imageSize - imgProp->height;
	int numBlocks = newImgSizePixel / 1024 + 1;

	removeSeam_ << <numBlocks, 1024 >> > (imgGray, idsToRemove->ids, imgProp, imgWithoutSeamGray);
	gpuErrchk(cudaDeviceSynchronize());

	imgProp->imageSize = newImgSizePixel;
	imgProp->width -= 1;
	
	updateImageGray_ << <newImgSizePixel / 1024 + 1, 1024 >> > (imgGray, imgWithoutSeamGray, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

}

void removePixelsFromSrc(pixel_t* imgSrc, pixel_t* imgWithoutSeamSrc, energyPixel_t* imgGray, imgProp_t* imgProp) {
	removePixelsFromSrc_ << <imgProp->imageSize / 1024 + 1, 1024 >> > (imgSrc, imgWithoutSeamSrc, imgGray, imgProp);
	gpuErrchk(cudaDeviceSynchronize());
}