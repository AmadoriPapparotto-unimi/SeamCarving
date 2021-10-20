#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>

static char SOURCE_PATH[100] = "src/assets/images/";

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
	pel_t color;
	float energy;
} energyPixel_t;

void readBMP(FILE* f, pixel_t* img, imgProp_t* imgProp);
void setupImgProp(imgProp_t* ip, FILE* f);
void writeBMP_pixel(char* p, pixel_t* img, imgProp_t* ip);
void writeBMP_energy(char* p, energyPixel_t* energyImg, imgProp_t* ip);
void toGrayScale(pixel_t* img, energyPixel_t* imgGray, imgProp_t* imgProp);
//void writeBMP_minimumSeam(char* p, energyPixel_t* energyImg, seam_t* minSeam, imgProp_t* imgProp);
void energy2pixel(pixel_t* img2convert, energyPixel_t* energyImg, imgProp_t* ip);
void writeBMPHeader(char* p, energyPixel_t* energyImg, imgProp_t* ip, int newSize);
void setBMP_header(imgProp_t* imgProp, int fileSize, int width);

#endif