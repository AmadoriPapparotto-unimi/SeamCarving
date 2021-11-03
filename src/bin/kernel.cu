#include "image_handler.h"
#include "seam_carving.h"
#include "utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys\timeb.h> 

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

char* src_path;

void applySeamCarving(char *p, int iterations) {

	/// <summary>
	/// Funzione principale Host che alloca la memoria necessaria all'algoritmo, legge l'immagine, lancia i vari step del seam carving e scrive la nuova immagine ottenuta.
	/// </summary>
	/// <param name="p">Il path dell'immagine in input</param>
	/// <param name="iterations">Il numero di pixel orizzontali da rimuovere su ogni riga</param>

	imgProp_t* imgProp;

	pixel_t* imgSrc;
	pixel_t* imgWithoutSeamSrc;
	
	energyPixel_t* imgGray;
	energyPixel_t* imgWithoutSeamGray;
	
	seam_t* seams;
	seam_t* minSeamsPerBlock;
	seam_t* minSeam;

	FILE* f = fopen(p, "rb");
	if (f == NULL) {
		printf("*** FILE NOT FOUND %s ***\n", p);
		exit(1);
	}

	// Si è deciso di utilizzare la cudaMallocManaged poichè è il metodo più nuovo che CUDA mette a disposizione.
	// Inoltre possedendo una GPU abbastanza nuova (1080ti) abbiamo potuto sfruttare tutte le varie ottimizzazione che tale api supporta in questa architettura.
	
	gpuErrchk(cudaMallocManaged(&imgProp, sizeof(imgProp_t)));

	//Si legge l'header dell'immagine e si estrapolano le caratteristiche utili
	setupImgProp(imgProp, f);

	int numBlocks = imgProp->width / 1024 + 1;

	gpuErrchk(cudaMallocManaged(&imgSrc, imgProp->imageSize * sizeof(pixel_t)));
	gpuErrchk(cudaMallocManaged(&imgGray, imgProp->imageSize * sizeof(energyPixel_t)));
	gpuErrchk(cudaMallocManaged(&imgWithoutSeamSrc, (imgProp->imageSize - (imgProp->height * iterations)) * sizeof(pixel_t)));
	gpuErrchk(cudaMallocManaged(&imgWithoutSeamGray, imgProp->imageSize * sizeof(energyPixel_t)));

	gpuErrchk(cudaMallocManaged(&seams, imgProp->width * sizeof(seam_t)));
	for (int i = 0; i < imgProp->width; i++)
		gpuErrchk(cudaMallocManaged(&seams[i].ids, imgProp->height * sizeof(int)));

	gpuErrchk(cudaMallocManaged(&minSeamsPerBlock, numBlocks * sizeof(seam_t)));
	for (int i = 0; i < numBlocks; i++)
		gpuErrchk(cudaMallocManaged(&minSeamsPerBlock[i].ids, imgProp->height * sizeof(int)));

	gpuErrchk(cudaMallocManaged(&minSeam, sizeof(seam_t)));
	gpuErrchk(cudaMallocManaged(&minSeam->ids, imgProp->height * sizeof(int)));
		
	//Si legge l'immagine
	readBMP(f, imgSrc, imgProp);
	struct timeb start, end;
	ftime(&start);
	//Si converte l'immagine in scala di grigi
	toGrayScale(imgSrc, imgGray, imgProp);
	
	//Si itera l'algoritmo di seam carving per il numero di iterazioni richieste dall'utente
	for (int i = 0; i < iterations; i++) {
		energyMap(imgGray, imgProp);	//si calcola la mappa dell'energia	
		findSeams(imgGray, imgSrc, imgProp, minSeam, seams, minSeamsPerBlock); // si trova il seam da rimuovere
		removeSeam(imgGray, imgWithoutSeamGray, minSeam, imgProp); // si rimuove il seam precedentemente trovato
		//printf("ITERAZIONE %d COMPLETATA\n", i);
	}

	// si rimuovono tutti i pixel nell'immagine a colori
	removePixelsFromSrc(imgSrc, imgWithoutSeamSrc, imgGray, imgProp);
	ftime(&end);
	int diff = (int)(1000.0 * (end.time - start.time)
		+ (end.millitm - start.millitm));;
	printf("\nOperation took %u milliseconds\n", diff);

	// si genera il nuovo header coerente con le caratteristiche dell'immagine finale
	setBMP_header(imgProp, 0, imgProp->width);

	//si scrive l'immagine finale
	writeBMP_pixel("C:\\aa\\reduced.bmp", imgWithoutSeamSrc, imgProp);
		
	fclose(f);
}

int main(int argc, char** argv) {

	// path dell'immagine 
	char* path = argv[1];
	// numero di iterazioni
	int iterations = atoi(argv[2]);
	int deviceID = 0;
	cudaDeviceProp props;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&props, deviceID);

	applySeamCarving(path, iterations);
	cudaDeviceReset();

	return 0;
}
