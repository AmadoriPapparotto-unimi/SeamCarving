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

__device__ int min(int id1, int id2, energyPixel_t* energyImg)
{
	return (energyImg[id1].energy < energyImg[id2].energy) ? id1 : id2;
}

__global__ void computeSeams(energyPixel_t* energyImg, seam_t* seams, imgProp_t* imgProp, bool colorSeams = false) {

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
			energyImg[currentId].pixel.R = 255;
			energyImg[currentId].pixel.B = 0;
			energyImg[currentId].pixel.G = 0;
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

		//seams[idThread].total_energy += energyImg[nextIdMin].energy;
		//if(i > 0)
		//	seams[idThread].ids[i] = nextIdMin;

	}

	//if (idThread == 0) {
	//	for (int i = 0; i < imgProp->height; i++) {
	//		printf("%d - ", seams[idThread].ids[i]);
	//	}
	//}
}
__global__ void removePixelPerRow_(energyPixel_t* energyImg, int idToRemove, energyPixel_t* newImage, int imageSize, int width, int idRow) {

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	int newPosition = idThread  + (width *idRow);
	int oldPosition = idThread + ((width+1) * idRow);

	if (idThread < width) {

		int shift = oldPosition < idToRemove ? 0 : 1;


		/*printf("idRow %d (%d)\n", idRow, newPosition);
		printf("idToRemove %d (%d)\n", idToRemove, newPosition);
		printf("shift %d (%d)\n", shift, newPosition);
		printf("new Position %d (%d)\n", newPosition + shift, newPosition);*/

		newImage[newPosition].energy = energyImg[oldPosition + shift].energy;
		newImage[newPosition].pixel.R = energyImg[oldPosition + shift].pixel.R;
		newImage[newPosition].pixel.G = energyImg[oldPosition + shift].pixel.G;
		newImage[newPosition].pixel.B = energyImg[oldPosition + shift].pixel.B;
	}


}

__global__ void removeSeam_(energyPixel_t* energyImg, seam_t* idsToRemove, energyPixel_t* newImage, imgProp_t* imgProp) {

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idThread < imgProp->height) {

		int idToRemove = idsToRemove->ids[idThread];
		removePixelPerRow_ << <(imgProp->width - 1) / 1024 + 1, 1024 >> > (energyImg, idToRemove, newImage, imgProp->imageSize, imgProp->width - 1, idThread);
		cudaDeviceSynchronize();

	}
	__syncthreads();
	cudaDeviceSynchronize();
}

void map(energyPixel_t* energyImg, imgProp_t* imgProp) {
	energyMap << <imgProp->imageSize / 1024 + 1, 1024 >> > (energyImg, imgProp);
	gpuErrchk(cudaDeviceSynchronize());
	//writeBMP_energy("src/assets/images/energy.bmp", energyImg, imgProp);
}



void findSeams(energyPixel_t* energyImg, imgProp_t* imgProp, seam_t* minSeam, seam_t* seams, seam_t* minSeamsPerBlock) {

	//energyPixel_t* img;
	//seam_t* seams;
	//seam_t* minSeamsPerBlock;


	int numBlocks = imgProp->width / 1024 + 1;


	//gpuErrchk(cudaMallocManaged(&img, imgProp->imageSize * sizeof(energyPixel_t)));


	//for (int i = 0; i < imgProp->imageSize; i++) {
	//	img[i].pixel.R = energyImg[i].energy;
	//	img[i].pixel.G = energyImg[i].energy;
	//	img[i].pixel.B = energyImg[i].energy;
	//	img[i].energy = energyImg[i].energy;
	//}


	computeSeams << <numBlocks, 1024 >> > (energyImg, seams, imgProp);
	//cudaDeviceSynchronize();

	/*for (int i = 0; i < imgProp->height; i++) {
		printf("%d - ", seams[0].ids[i]);
	}*/

	gpuErrchk(cudaDeviceSynchronize());

	//pixel_t* img2convert = (pixel_t*)malloc(imgProp->imageSize * sizeof(pixel_t));
	//energy2pixel(img2convert, img, imgProp);
	//writeBMP_pixel(strcat(SOURCE_PATH,"seams_map.bmp"), img2convert, imgProp);

	//minArr(numBlocks, 1024, seams, minSeamsPerBlock, imgProp);
	//gpuErrchk(cudaDeviceSynchronize());

	//for (int i = 0; i < imgProp->height; i++) {
	//	printf("%d - ", minSeamsPerBlock[0].ids[i]);
	//}

	//printf("\n\n");

	//for (int i = 0; i < imgProp->height; i++) {
	//	printf("%d - ", minSeamsPerBlock[1].ids[i]);
	//}

	//minArr(1, imgProp->width / 1024 + 1, minSeamsPerBlock, minSeam, imgProp);
	//gpuErrchk(cudaDeviceSynchronize());
	dummyMin(seams, *minSeam, imgProp);

	//for (int i = 0; i < imgProp->height; i++) {
	//	printf("%d - ", minSeam[0].ids[i]);
	//}

	//minSeam->total_energy = seams[0].total_energy;
	//for (int i = 0; i < imgProp->height; i++) {
	//	
	//	minSeam->ids[i] = seams[0].ids[i];

	//}


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
	//gpuErrchk(cudaDeviceSynchronize());
	//gpuErrchk(cudaFree(minSeamsPerBlock));

}

void removeSeam(energyPixel_t* imgGray, seam_t* idsToRemove, imgProp_t* imgProp) {
	energyPixel_t* imgWithoutSeam;


	int newImgSizePixel = imgProp->imageSize - imgProp->height;
	int newFileSize = newImgSizePixel * 3 + 54;
	int numBlocks = newImgSizePixel / 1024 + 1;
	gpuErrchk(cudaMallocManaged(&imgWithoutSeam, newImgSizePixel * sizeof(energyPixel_t)));

	//removeSeam_ << <numBlocks, 1024 >> > (imgGray, idsToRemove, imgWithoutSeam, imgProp->imageSize, imgProp->width);

	removeSeam_ << <imgProp->height/1024 +1, 1024 >> > (imgGray, idsToRemove, imgWithoutSeam, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

	imgProp->imageSize = newImgSizePixel;
	imgProp->width -= 1;
	for (int i = 0; i < imgProp->imageSize; i++) {
		imgGray[i].energy = imgWithoutSeam[i].energy;
		imgGray[i].pixel.R = imgWithoutSeam[i].pixel.R;
		imgGray[i].pixel.G = imgWithoutSeam[i].pixel.G;
		imgGray[i].pixel.B = imgWithoutSeam[i].pixel.B;
	}
	gpuErrchk(cudaFree(imgWithoutSeam));
	//writeBMPHeader(strcat(SOURCE_PATH, "without_seam.bmp"), imgGray, imgProp, totalSizeNewImage);

	//cudaFree(&minSeam[0].ids);
	//cudaFree(minSeam);

}