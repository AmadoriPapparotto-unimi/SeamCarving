#include "image_handler.h"
#include "seam_carving.h"
#include "math.h"

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

	if (id % width == 0) {
		if (id == 0)
			return 0;
		if (id == imageSize - width)
			return 1;
		return 2;
	}
	else if (id + 1 % width == 0) {
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

	char pos = getPosition(id, imgProp);
	switch (pos)
	{
	case 0:
		dx2 = powf(energyPixel[id + 1].pixel.R, 2);
		dy2 = powf(energyPixel[id + imgProp->width].pixel.R, 2);
		break;
	case 1:
		dx2 = powf(energyPixel[id + 1].pixel.R, 2);
		dy2 = powf(energyPixel[id - imgProp->width].pixel.R, 2);
		break;
	case 2:
		dx2 = powf(energyPixel[id + 1].pixel.R, 2);
		dy2 = powf(energyPixel[id + imgProp->width].pixel.R - energyPixel[id - imgProp->width].pixel.R, 2);
		break;
	case 3:
		dx2 = powf(energyPixel[id - 1].pixel.R, 2);
		dy2 = powf(energyPixel[id + imgProp->width].pixel.R, 2);
		break;
	case 4:
		dx2 = powf(energyPixel[id - 1].pixel.R, 2);
		dy2 = powf(energyPixel[id - imgProp->width].pixel.R, 2);
		break;
	case 5:
		dx2 = powf(energyPixel[id - 1].pixel.R, 2);
		dy2 = powf(energyPixel[id + imgProp->width].pixel.R - energyPixel[id - imgProp->width].pixel.R, 2);
		break;
	case 6:
		dx2 = powf(energyPixel[id - 1].pixel.R - energyPixel[id + 1].pixel.R, 2);
		dy2 = powf(energyPixel[id + imgProp->width].pixel.R, 2);
		break;
	case 7:
		dx2 = powf(energyPixel[id - 1].pixel.R - energyPixel[id + 1].pixel.R, 2);
		dy2 = powf(energyPixel[id - imgProp->width].pixel.R, 2);
		break;
	default:
		dx2 = powf(energyPixel[id - 1].pixel.R - energyPixel[id + 1].pixel.R, 2);
		dy2 = powf(energyPixel[id + imgProp->width].pixel.R - energyPixel[id - imgProp->width].pixel.R, 2);
		break;
	}
	//__syncthreads();

	pixel->energy = sqrtf((double)dx2 + dy2);
}

__global__ void energyMap(energyPixel_t* energyImg, imgProp_t* imgProp) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < imgProp->imageSize) {
		calculateEnergy(energyImg, &energyImg[id], id, imgProp);
	}
}

void map(energyPixel_t* energyImg, imgProp_t* imgProp) {
	energyMap << <imgProp->imageSize / 1024, 1024 >> > (energyImg, imgProp);
	cudaDeviceSynchronize();
	writeBMP_energy("src/assets/images/energy.bmp", energyImg, imgProp);
}

__device__ int min(int id1, int id2, energyPixel_t* energyImg)
{
	return (energyImg[id1].energy < energyImg[id2].energy) ? id1 : id2;
}

__global__ void computeSeams(energyPixel_t* energyImg, imgProp_t* imgProp) {

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

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id > imgProp->width)
		return;
	int id_min = 0;

	int total_energy = energyImg[id].energy;

	energyImg[id].pixel.R = 255;
	energyImg[id].pixel.B = 0;
	energyImg[id].pixel.G = 0;

	for (int i = 0;i< imgProp->height-2;i++) {		

		int pos = getPosition(id, imgProp);
		switch (pos)
		{
		case 0:
		case 2:
			id_min= min(id + imgProp->width, id + 1 + imgProp->width, energyImg);
			break;
		case 3:
		case 5:
			id_min = min(id + imgProp->width,id - 1 + imgProp->width, energyImg);
			break;
		default:
			id_min =  min(min(id + imgProp->width, id - 1 + imgProp->width, energyImg),
				id + 1 + imgProp->width, energyImg);
			break;
		}

		total_energy += energyImg[id_min].energy;
		
		id = id_min;


		energyImg[id_min].pixel.R = 255;
		energyImg[id_min].pixel.B = 0;
		energyImg[id_min].pixel.G = 0;
		

	}
	__syncthreads();

	if (id == 0) {
	
	}
}

void findSeams(energyPixel_t* energyImg, imgProp_t* imgProp) {

	energyPixel_t* img;
	cudaMallocManaged(&img, imgProp->imageSize * sizeof(energyPixel_t));

	for (int i = 0; i < imgProp->imageSize; i++) {
		img[i].pixel.R = energyImg[i].energy;
		img[i].pixel.G = energyImg[i].energy;
		img[i].pixel.B = energyImg[i].energy;
		img[i].energy = energyImg[i].energy;
	}
	//printf("%f %d %d %d\n", energyImg[0].energy, energyImg[0].pixel.R, energyImg[0].pixel.B, energyImg[0].pixel.G);




	//writeBMP_pixel(strcat(SOURCE_PATH, "energy_seams.bmp"), energy2pixel(img, imgProp), imgProp);

	computeSeams << <imgProp->width/1024+1, 1024 >> > (img, imgProp);
	cudaDeviceSynchronize();
	writeBMP_pixel(strcat(SOURCE_PATH,"seams_map.bmp"), energy2pixel(img, imgProp), imgProp);

}

