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

#define IMAGESIZE 


pel** ReadBMP() {
	FILE* f = fopen("source/assets/images/castle_bmp.bmp", "rb");
	if (f == NULL) {
		printf("\n\n NOT FOUND\n\n");
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

	printf("\n   Input BMP File name: %20s  (%u x %u)", "immagine", ip.Hpixels, ip.Vpixels);

	pel tmp;

	pel** TheImage;

	cudaMallocManaged(&TheImage, height * sizeof(pel*));
	for (unsigned int i = 0; i < height; i++)
		cudaMallocManaged(&TheImage[i], RowBytes * sizeof(pel));

	for (unsigned int i = 0; i < height; i++) {
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);
		if (i >= height - 4) {
			printf("r %d", TheImage[i][0]);
		}
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

}