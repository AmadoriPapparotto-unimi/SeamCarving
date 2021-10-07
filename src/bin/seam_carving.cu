#include "image_handler.h"
#include "seam_carving.h"
#include "math.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

__device__ char getPosition(int id, imgProp_t* imgProp) {
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

__device__ void calculateEnergy(energyPixel_t* energyPixel, energyPixel_t* pixel, int id, imgProp_t* imgProp) {
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
	int inde = 1382303 - (1428) - (1428 - 540);

	char pos = getPosition(id, imgProp);
	switch (pos)
	{
	case 0:
		dx2 = energyPixel[id + 1].pixel.R;
		dy2 = energyPixel[id + imgProp->width].pixel.R;
		break;
	case 1:
		dx2 = energyPixel[id + 1].pixel.R;
		dy2 = energyPixel[id - imgProp->width].pixel.R;
		break;
	case 2:
		dx2 = energyPixel[id + 1].pixel.R;
		dy2 = energyPixel[id + imgProp->width].pixel.R - energyPixel[id - imgProp->width].pixel.R;
		break;
	case 3:
		dx2 = energyPixel[id - 1].pixel.R;
		dy2 = energyPixel[id + imgProp->width].pixel.R;
		break;
	case 4:
		dx2 = energyPixel[id - 1].pixel.R;
		dy2 = energyPixel[id - imgProp->width].pixel.R;
		break;
	case 5:
		dx2 = energyPixel[id - 1].pixel.R;
		dy2 = energyPixel[id + imgProp->width].pixel.R - energyPixel[id - imgProp->width].pixel.R;
		break;
	case 6:
		dx2 = energyPixel[id - 1].pixel.R - energyPixel[id + 1].pixel.R;
		dy2 = energyPixel[id + imgProp->width].pixel.R;
		break;
	case 7:
		dx2 = energyPixel[id - 1].pixel.R - energyPixel[id + 1].pixel.R;
		dy2 = energyPixel[id - imgProp->width].pixel.R;
		break;
	case -1:
		dx2 = energyPixel[id - 1].pixel.R - energyPixel[id + 1].pixel.R;
		dy2 = energyPixel[id + imgProp->width].pixel.R - energyPixel[id - imgProp->width].pixel.R;
		break;
	}

	pixel->energy = sqrtf((dx2 * dx2) + (dy2 * dy2));
}

__global__ void energyMap(energyPixel_t* energyImg, imgProp_t* imgProp) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imgProp->imageSize) {
		calculateEnergy(energyImg, &energyImg[id], id, imgProp);
	}
}

void map(energyPixel_t* energyImg, imgProp_t* imgProp) {
	energyMap << <imgProp->imageSize / 1024 + 1, 1024 >> > (energyImg, imgProp);
	cudaDeviceSynchronize();
	writeBMP_energy("src/assets/images/energy.bmp", energyImg, imgProp);
}

__device__ int min(int id1, int id2, energyPixel_t* energyImg)
{
	return (energyImg[id1].energy < energyImg[id2].energy) ? id1 : id2;
}

__global__ void computeSeams(energyPixel_t* energyImg, seam_t* seams, imgProp_t* imgProp) {

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
		energyImg[currentId].pixel.R = 255;
		energyImg[currentId].pixel.B = 0;
		energyImg[currentId].pixel.G = 0;

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

		//seams[idThread].total_energy += energyImg[nextIdMin].energy;
		//if(i > 0)
		//	seams[idThread].ids[i] = nextIdMin;

	}
	__syncthreads();
}

void findSeams(energyPixel_t* energyImg, imgProp_t* imgProp) {

	energyPixel_t* img;
	seam_t* seams;
	seam_t* minSeamsPerBlock;
	seam_t* minmin;

	int numBlocks = imgProp->width / 1024 + 1;

	cudaMallocManaged(&img, imgProp->imageSize * sizeof(energyPixel_t));

	cudaMallocManaged(&seams, imgProp->width * sizeof(seam_t));
	for (int i = 0; i < imgProp->width; i++)
		cudaMallocManaged(&seams[i].ids, imgProp->height * sizeof(int));

	cudaMallocManaged(&minSeamsPerBlock, numBlocks * sizeof(seam_t));
	for (int i = 0; i < numBlocks; i++)
		cudaMallocManaged(&minSeamsPerBlock[i].ids, imgProp->height * sizeof(int));

	cudaMallocManaged(&minmin, sizeof(seam_t));
	cudaMallocManaged(&minmin[0].ids, imgProp->height * sizeof(int));

	for (int i = 0; i < imgProp->imageSize; i++) {
		img[i].pixel.R = energyImg[i].energy;
		img[i].pixel.G = energyImg[i].energy;
		img[i].pixel.B = energyImg[i].energy;
		img[i].energy  = energyImg[i].energy;
	}
	cudaDeviceSynchronize();
	computeSeams << <numBlocks, 1024 >> > (img, seams, imgProp);
	cudaDeviceSynchronize();
	writeBMP_pixel(strcat(SOURCE_PATH,"seams_map.bmp"), energy2pixel(img, imgProp), imgProp);

	min(numBlocks, 1024, seams, minSeamsPerBlock, imgProp);
	cudaDeviceSynchronize();

	min(1, imgProp->width / 1024 + 1, minSeamsPerBlock, minmin, imgProp);
	cudaDeviceSynchronize();

	for (int y = 0; y < imgProp->height; y++) {
		img[minmin[0].ids[y]].pixel.R = 0;
		img[minmin[0].ids[y]].pixel.G = 255;
		img[minmin[0].ids[y]].pixel.B = 0;
	}
	writeBMP_pixel(strcat(SOURCE_PATH, "seams_map_minimum.bmp"), energy2pixel(img, imgProp), imgProp);
	//writeBMP_minimumSeam(strcat(SOURCE_PATH, "seams_map.bmp"), img, minmin, imgProp);

	printf("%d", minmin[0].total_energy);
}