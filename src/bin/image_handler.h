#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED

static char SOURCE_PATH[100] = "src/assets/images/";

typedef struct ImgPropStruct {
	int width;
	int height;
	unsigned char headerInfo[54];
	int imageSize;
} imgProp_t;

typedef unsigned char pel_t;

typedef struct PixelStruct {
	pel_t B;
	pel_t G;
	pel_t R;
} pixel_t;

typedef struct EnergyPixelStruct {
	pixel_t pixel;
	double energy;
} energyPixel_t;

void readBMP(pixel_t* img, energyPixel_t* imgGray, char* p, imgProp_t* ip);
void writeBMP_pixel(char* p, pixel_t* img, imgProp_t* ip);
void writeBMP_energy(char* p, energyPixel_t* energyImg, imgProp_t* ip);
//void writeBMP_minimumSeam(char* p, energyPixel_t* energyImg, seam_t* minSeam, imgProp_t* imgProp);
pixel_t* energy2pixel(energyPixel_t* energyImg, imgProp_t* ip);

#endif