#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED

typedef struct ImgPropStruct {
	int width;
	int height;
	unsigned char headerInfo[54];
	unsigned long int rowBytes;
} ImgProp;

struct Pixel {
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

typedef unsigned char pel;

pel** ReadBMP(char* p);
//void setupImgProp(ImgProp* ip, FILE* f);

#endif