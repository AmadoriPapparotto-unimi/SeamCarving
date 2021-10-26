#include "image_handler.h"
#include "seam_carving.h"
#include "utils.cuh"
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

__device__ 
void grayValue(energyPixel_t *energyPixel, pel_t r, pel_t g, pel_t b, int id) {

	/// <summary>
	
	/// Funione GPU che si occupa di convertire l'immagine a colori in scala di grigi
	
	/// </summary>
	/// <param name="energyPixel">L'immagine da convertire</param>
	/// <param name="r">Valore ROSSO del pixel id</param>
	/// <param name="g">Valore VERDE del pixel id</param>
	/// <param name="b">Valore BLUE del pixel id</param>
	/// <param name="id">Id del pixel da convertire</param>
	
	int grayVal = (r + g + b) / 3;
	energyPixel->color = grayVal;
	energyPixel->id_pixel = id;
}

__global__ 
void toGrayScale_(pixel_t* img, energyPixel_t* imgGray, int imageSize)
{
	/// <summary>
	/// Kernel GPU per applicare il calcolo del grayscale su ogni pixel
	/// </summary>
	/// <param name="img">Immagine (input)</param>
	/// <param name="imgGray">Immagine in scala di grigi (output)</param>
	/// <param name="imageSize">Numero totale di pixel dell'immagine</param>
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < imageSize) {
		grayValue(&imgGray[id], img[id].R, img[id].G, img[id].B, id);
	}
}

__global__
void generateEnergyImg_(pixel_t* imgSrc, energyPixel_t* energyImg, imgProp_t* imgProp) {
	/// <summary>
	/// Kernel GPU per generare una immagine di tipo pixel_t raffigurante la mappa di energia a partire dall'immagine in scala di grigi
	/// </summary>
	/// <param name="imgSrc">Immagine (output)</param>
	/// <param name="energyImg">Immagine con mappa di energia (input)</param>
	/// <param name="imgProp">Proprieta' dell'immagine</param>
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgSrc[idThread].R = energyImg[idThread].energy;
		imgSrc[idThread].G = energyImg[idThread].energy;
		imgSrc[idThread].B = energyImg[idThread].energy;
	}
}

__global__
void energy2pixel_(pixel_t* imgSrc, energyPixel_t* energyImg, imgProp_t* imgProp) {
	/// <summary>
	/// Kernel di supporto non utilizzato nell'algoritmo. Serve per convertire una immagine di tipo energyPixel_t in una pixel_t.
	/// E' stata utilizzata per ottenere le immagini delle fasi intermedie dell'algoritmo
	/// </summary>
	/// <param name="imgSrc"> Immagine di output</param>
	/// <param name="energyImg"> Immagine input da convertire</param>
	/// <param name="imgProp"> Proprietà della imamgine</param>

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgSrc[idThread].R = energyImg[idThread].color;
		imgSrc[idThread].G = energyImg[idThread].color;
		imgSrc[idThread].B = energyImg[idThread].color;
	}
}

__global__
void colorSeamToRemove_(pixel_t* img, seam_t* seam, imgProp_t* imgProp) {
	/// <summary>
	/// Kernel GPU per colorare un seam dell'immagine
	/// </summary>
	/// <param name="img">Immagine (output)</param>
	/// <param name="seam">Seam da colorare (input)</param>
	/// <param name="imgProp">Proprieta' dell'immagine</param>
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		img[seam->ids[idThread]].R = 0;
		img[seam->ids[idThread]].G = 255;
		img[seam->ids[idThread]].B = 0;
	}
}

void toGrayScale(pixel_t* img, energyPixel_t* imgGray, imgProp_t* imgProp) {
	/// <summary>
	/// Funzione host che richiama il kernel GPU per la colorazione in scala di grigi
	/// </summary>
	/// <param name="img">Immagine (input)</param>
	/// <param name="imgGray">Immagine in scala di grigi (output)</param>
	/// <param name="imgProp">Proprieta' dell'immagine</param>
	int blocks = imgProp->imageSize / 1024 + 1;

	toGrayScale_<< <blocks, 1024 >> > (img, imgGray, imgProp->imageSize);
	gpuErrchk(cudaDeviceSynchronize());
}

void setupImgProp(imgProp_t* imgProp, FILE* f) {
	/// <summary>
	/// Funzione host che permette di estrapolare le proprietà dell'immagine di input mediante la lettura del suo header
	/// </summary>
	/// <param name="imgProp">Output</param>
	/// <param name="f">File dell'immagine di input</param>
	pel_t headerInfo[54];
	fread(headerInfo, sizeof(pel_t), 54, f);


	int width = *(int*)&headerInfo[18]; // l'indice 18 dell'header contiene la larghezza dell'immagine(in pixel)
	int height = *(int*)&headerInfo[22]; // l'indice 22 dell'header  contiene l'altezza dell'immagine (in pixel)
	printf("#bytes: %d\n", *(int*)&headerInfo[34]); // l'indice 34 dell'header contiene la grandezza dell'immagine in byte

	for (unsigned int i = 0; i < 54; i++)
		imgProp->headerInfo[i] = headerInfo[i];

	imgProp->height = height;
	imgProp->width = width;
	imgProp->imageSize = width * height;

	printf("Input BMP dimension: (%u x %u)\n", imgProp->width, imgProp->height);
	printf("IHeader[2] %d\n", *(int*)&headerInfo[2]);
}

void readBMP(FILE *f, pixel_t* img, imgProp_t* imgProp) {

	/// <summary>
	/// Legge un file BMP tenendo conto e conseguentemente scartando il padding all'interno di esso.
	/// Il formato BMP prevede l'inserimento di N byte a fine riga in modo tale che ogni indirizzo di inizio riga sia allineato mod 4 byte.
	/// </summary>
	/// <param name="f">File BMP</param>
	/// <param name="img">Immagine</param>
	/// <param name="imgProp">Proprieta' dell'immagine</param>

	//img[0] = B
	//img[1] = G
	//img[2] = R
	//BMP LEGGE I PIXEL NEL FORMATO BGR
	for (int r = 0; r < imgProp->height; r++) {
		fread(&img[r*imgProp->width], sizeof(pel_t), imgProp->width * sizeof(pixel_t), f);

		int padding = 4 - ((imgProp->width * 3) % 4);
		if (padding != 0 && padding != 4) {
			fseek(f, padding, SEEK_CUR);
			
		}
	}

}

void writeBMP_pixel(char* p, pixel_t* img, imgProp_t* ip) {
	
	/// <summary>
	/// Scrittura di un file BMP, che tiene conto del numero di byte padding da aggiungere.
	/// </summary>
	/// <param name="p">Path del file</param>
	/// <param name="img">Immagine da scrivere</param>
	/// <param name="ip">Proprieta' dell'immagine</param>

	FILE* fw = fopen(p, "wb");

	printf("FINAL HEIGHT %d\n", ip->height);
	printf("FINAL WIDTH %d\n", ip->width);
	fwrite(ip->headerInfo, 1, 54, fw);

	int padding = 0;
	for (int r = 0; r < ip->height; r++) {
		for (int c = 0; c < ip->width; c++) {
			fputc(img[c + r * ip->width].B, fw);
			fputc(img[c + r * ip->width].G, fw);
			fputc(img[c + r * ip->width].R, fw);
		}
		padding = 4 - ((ip->width * 3) % 4);
		if (padding != 0 && padding != 4) {
			for (int i = 0; i < padding; i++) {
				fputc(0, fw);
			}
		}
	}
	fflush(fw);

	fclose(fw);
	printf("Immagine %s generata\n", p);
}

void writeBMP_energy(char* p, energyPixel_t* energyImg, imgProp_t* imgProp) {
	/// <summary>
	/// Scrittura di un file BMP a partire da una immagine di tipo energyPixel_t. Funzione non utilizzata dall'algoritmo, ma necessaria ad ottenere le stampe intermedie
	/// </summary>
	/// <param name="p">Path del file</param>
	/// <param name="energyImg">Immagine energyPixel_t da scrivere</param>
	/// <param name="imgProp">Prorpieta' dell'immagine</param>
	pixel_t* img;
	
	// riserva lo spazio per la nuova immagine da generare (convertita)
	cudaMallocManaged(&img, imgProp->imageSize * sizeof(pixel_t)); 

	//kernel che genera l'immagine convertita
	generateEnergyImg_ << <imgProp->imageSize/1024 + 1, 1024>> > (img, energyImg, imgProp); 
	gpuErrchk(cudaDeviceSynchronize());

	//scrittura dell'immagine su file
	writeBMP_pixel(p, img, imgProp); 

	gpuErrchk(cudaFree(img));
}

void writeBMP_grayscale(energyPixel_t* imgGray, imgProp_t* imgProp) {
	/// <summary>
	/// Scrittura di un file BMP a partire da una immagine di tipo energyPixel_t (raffigurante l'immagine in scala di grigi). Funzione non utilizzata dall'algoritmo, ma necessaria ad ottenere le stampe intermedie
	/// </summary>
	/// <param name="imgGray">Immagine in scala di grigi</param>
	/// <param name="imgProp">Proprieta' dell'immagine</param>
	pixel_t* img2convert;


	// riserva lo spazio per la nuova immagine da generare (convertita)
	cudaMallocManaged(&img2convert, imgProp->imageSize * sizeof(pixel_t));

	//kernel che genera l'immagine convertita
	energy2pixel_ << <imgProp->imageSize/1024 + 1, 1024 >> > (img2convert, imgGray, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

	//scrittura dell'immagine su file
	writeBMP_pixel("C:/aa/gray.bmp", img2convert, imgProp);
	gpuErrchk(cudaFree(img2convert));
}

void writeBMP_minimumSeam(char* p, pixel_t* img, seam_t* minSeam, imgProp_t* imgProp) {
	/// <summary>
	/// Scrittura di un file BMP con colorazione del minSeam. Funzione non utilizzata dall'algoritmo, ma necessaria ad ottenere le stampe intermedie
	/// </summary>
	/// <param name="p"></param>
	/// <param name="img"></param>
	/// <param name="minSeam"></param>
	/// <param name="imgProp"></param>

	// riserva lo spazio per la nuova immagine da generare (con minimumSeam colorato)
	colorSeamToRemove_ << <imgProp->height/1024 + 1, 1024>> > (img, minSeam, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

	//scrittura dell'immagine su file
	writeBMP_pixel("C:/aa/seams_map_minimum.bmp", img, imgProp);
	gpuErrchk(cudaFree(img));
}

void setBMP_header(imgProp_t* imgProp, int fileSize, int width) {
	/// <summary>
	/// Setup dell'header di un file BMP per i seguenti parametri:
	/// - dimensione totale del file
	/// - altezza (in pixel) dell'immagine
	/// </summary>
	/// <param name="imgProp">Proprieta' dell'immagine</param>
	/// <param name="fileSize">Dimensione (in byte) del file</param>
	/// <param name="width"> Larghezza dell'immagine </param>
	imgProp->headerInfo[2] = (unsigned char)(fileSize >> 0) & 0xff;
	imgProp->headerInfo[3] = (unsigned char)(fileSize >> 8) & 0xff;
	imgProp->headerInfo[4] = (unsigned char)(fileSize >> 16) & 0xff;
	imgProp->headerInfo[5] = (unsigned char)(fileSize >> 24) & 0xff;
	
	imgProp->headerInfo[18] = (unsigned char)(width >> 0) & 0xff;
	imgProp->headerInfo[19] = (unsigned char)(width >> 8) & 0xff;
	imgProp->headerInfo[20] = (unsigned char)(width >> 16) & 0xff;
	imgProp->headerInfo[21] = (unsigned char)(width >> 24) & 0xff;
}