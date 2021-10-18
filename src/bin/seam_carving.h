#pragma once
#include "image_handler.h";
#include "image_handler.h";

typedef struct seamStruct {
	int total_energy;
	int* ids;
} seam_t;

void map(energyPixel_t* energyImg, imgProp_t* imgProp);
void findSeams(energyPixel_t* energyImg, imgProp_t* imgProp, seam_t* minSeam);
void removeSeam(energyPixel_t* imgGray, seam_t* idsToRemove, imgProp_t* imgProp);