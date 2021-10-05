#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED

static char SOURCE_PATH[100] = "src/assets/images/";

typedef struct ImgPropStruct {
	int width;
	int height;
	unsigned char headerInfo[54];
	int imageSize;
} imgProp;

typedef unsigned char pel;

typedef struct PixelStruct {
	pel B;
	pel G;
	pel R;
} pixel;

typedef struct EnergyPixelStruct {
	pixel pixel;
	float energy;
} energyPixel;

pixel* readBMP(char* p);
void writeBMP_pixel(char* p, imgProp imgProp, pixel* img);
void writeBMP_pel(char* p, imgProp imgProp, pel* img);
pixel* energy2pixel(imgProp imgProp, energyPixel* energyImg);

#endif