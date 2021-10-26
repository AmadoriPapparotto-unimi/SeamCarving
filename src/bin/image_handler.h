#ifndef IMAGEHANDLER_H_INCLUDED
#define IMAGEHANDLER_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>

typedef struct ImgPropStruct {
	/// <summary>
	/// Struttura che contiene le proprietà di una immagine. Larghezza, altezza e dimensione in byte, tutte ottenute dall'header BMP. 
	/// headerInfo contiene l'header dell'immagine
	/// </summary>

	int width; //l'arghezza dell'immagine
	int height; //altezza dell'immagine
	int imageSize; //grandezza in byte dell'immagine
	unsigned char headerInfo[54]; //header BMP dell'immagine

} imgProp_t;

typedef unsigned char pel_t;

typedef struct PixelStruct {
	/// <summary>
	/// Rappresenta un pixel dell'immagine
	/// </summary>

	pel_t B;
	pel_t G;
	pel_t R;

} pixel_t;

typedef struct EnergyPixelStruct {
	/// <summary>
	/// Rappresenta un pixel di una immagine con energia. Si è scelto di usare long long per idPixel per due motivi:
	/// 1) E' possibile indicizzare immagini più grandi.
	/// 2) Migliora le performance di utilizzo della memoria.
	/// </summary>
	
	long long idPixel; //idOriginale dell'immagine iniziale
	float energy; //energia del pixel
	pel_t color; //il colore in scala di grigi

} energyPixel_t;

typedef struct seamStruct {

	/// <summary>
	///  Struttura che rappresenta un seam.
	/// Un seam è una path di pixel che collega il bordo inferiore a quello superiore.
	/// Il pixel successivo appartenete al path sarà quello minore tra i vicini(superiori) del pixel preso in considerazione. 
	/// </summary>

	float total_energy; //enerigia totale del seam
	int* ids; // id del seam, che corrisponde al pixel di partenza

} seam_t;

void readBMP(FILE* f, pixel_t* img, imgProp_t* imgProp);
void setupImgProp(imgProp_t* ip, FILE* f);
void toGrayScale(pixel_t* img, energyPixel_t* imgGray, imgProp_t* imgProp);

void setBMP_header(imgProp_t* imgProp, int fileSize, int width);

void writeBMP_minimumSeam(char* p, pixel_t* img, seam_t* minSeam, imgProp_t* imgProp);
void writeBMP_pixel(char* p, pixel_t* img, imgProp_t* ip);
void writeBMP_energy(char* p, energyPixel_t* energyImg, imgProp_t* ip);
void writeBMP_grayscale(energyPixel_t* imgGray, imgProp_t* imgProp);

#endif