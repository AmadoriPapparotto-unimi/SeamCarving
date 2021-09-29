#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED

typedef struct ImgPropStruct {
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[54];
	unsigned long int Hbytes;
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