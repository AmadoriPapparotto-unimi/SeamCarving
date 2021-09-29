#include "imageHandler.h"
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"


int main(int argc, char** argv) {

	pel** imgSrc;				

	//imgSrc = ReadBMP(argv[1]);
	imgSrc = ReadBMP("assets/images/castle_bmp.bmp");

	if (imgSrc == NULL) {
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	return 0;
}