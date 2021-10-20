#pragma once
#include "image_handler.h";

typedef struct seamStruct {
	int total_energy;
	int* ids;
} seam_t;

void energyMap(energyPixel_t* energyImg, imgProp_t* imgProp);
void findSeams(energyPixel_t* energyImg, pixel_t* imgSrc, imgProp_t* imgProp, seam_t* minSeam, seam_t* seams, seam_t* minSeamsPerBlock);
void removeSeam(energyPixel_t* imgGray, pixel_t* imgSrc, energyPixel_t* imgWithoutSeamGray, pixel_t* imgWithoutSeamSrc, seam_t* idsToRemove, imgProp_t* imgProp);