#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"


struct ImgProp {
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[54];
	unsigned long int Hbytes;
};

struct Pixel {
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

typedef unsigned char pel;    

pel** ReadBMP(char*);  
struct ImgProp ip;

pel** ReadBMP() {

	//BMP LEGGE I PIXEL NEL FORMATO BGR
	FILE* f = fopen("src/assets/images/castle_bmp.bmp", "rb");
	if (f == NULL) {
		printf("\n\nNOT FOUND\n\n");
		exit(1);
	}

	pel HeaderInfo[54];
	fread(HeaderInfo, sizeof(pel), 54, f); 
	
	int width = *(int*)&HeaderInfo[18];
	int height = *(int*)&HeaderInfo[22];
	
	for (unsigned int i = 0; i < 54; i++)
		ip.HeaderInfo[i] = HeaderInfo[i];

	ip.Vpixels = height;
	ip.Hpixels = width;
	int RowBytes = (width * 3 + 3) & (~3);
	ip.Hbytes = RowBytes;

	printf("\nInput BMP File name: (%u x %u; %u)", ip.Hpixels, ip.Vpixels, ip.Hbytes);

	pel** TheImage;

	cudaMallocManaged(&TheImage, height * sizeof(pel*));
	for (unsigned int i = 0; i < height; i++)
		cudaMallocManaged(&TheImage[i], RowBytes * sizeof(pel));

	for (unsigned int i = 0; i < height; i++) {
		fread(TheImage[i], sizeof(pel), RowBytes, f);
	}

	fclose(f);
	return TheImage;  // remember to free() it in caller!
}

int main(int argc, char** argv) {

	pel** imgSrc;				

	imgSrc = ReadBMP();

	if (imgSrc == NULL) {
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	return 0;
}