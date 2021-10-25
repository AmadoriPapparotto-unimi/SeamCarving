#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>

typedef struct ImgPropStruct {
	int width;
	int height;
	int imageSize;
	unsigned char headerInfo[54];
} imgProp_t;

typedef unsigned char pel_t;

typedef struct PixelStruct {
	pel_t B;
	pel_t G;
	pel_t R;
} pixel_t;

typedef struct EnergyPixelStruct {
	long long idPixel; //idOriginale dell'immagine iniziale
	float energy;
	pel_t color;
} energyPixel_t;

typedef struct seamStruct {
	float total_energy;
	int* ids;
} seam_t;

void readBMP(FILE* f, pixel_t* img, imgProp_t* imgProp);
void setupImgProp(imgProp_t* ip, FILE* f);
void toGrayScale(pixel_t* img, energyPixel_t* imgGray, imgProp_t* imgProp);

void setBMP_header(imgProp_t* imgProp, int fileSize, int width);

void writeBMP_minimumSeam(char* p, pixel_t* img, seam_t* minSeam, imgProp_t* imgProp);
void writeBMP_pixel(char* p, pixel_t* img, imgProp_t* ip);
void writeBMP_energy(char* p, energyPixel_t* energyImg, imgProp_t* ip);
void writeBMP_grayscale(energyPixel_t* imgGray, imgProp_t* imgProp);

#endif