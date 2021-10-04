#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED

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

pixel* readBMP(char* p);
void writeBMP_pixel(char* p, imgProp imgProp, pixel* img);
void writeBMP_pel(char* p, imgProp imgProp, pel* img);

#endif