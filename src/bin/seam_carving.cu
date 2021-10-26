#include "image_handler.h"
#include "seam_carving.h"
#include "math.h"
#include "utils.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

__device__ 
char getPosition(int id, imgProp_t* imgProp) {

	/// <summary>
	/// Funzione device che restituisce la posizione del pixel all'interno dell'immagine, ovvero se bordo, angolo o centrale.
	/// ANGOLO BASSO SX = 0
	/// ANGOLO ALTO SX = 1
	/// COLONNA SX = 2
	/// ANGOLO BASSO DX = 3
	/// ANGOLO ALTO DX = 4
	/// COLONNA DX = 5
	/// RIGA DOWN = 6
	/// RIGA UP = 7
	/// IN MEZZO
	/// </summary>
	/// <param name="id">Id del pixel che si sta considerando</param>
	/// <param name="imgProp">Proprietà dell'immagine</param>
	/// <returns>Il tipo di posizione del pixel all'interno dell'immagine</returns>


	//ANGOLO BASSO SX:									[0]
	//ANGOLO ALTO SX									[1]
	//COLONNA SX: id % imgProp->width == 0				[2]
	//ANGOLO BASSO DX									[3]
	//ANGOLO ALTO DX									[4]
	//COLONNA DX: id + 1 %  imgProp->width == 0			[5]
	//RIGA DOWN: id < imgProp->width					[6]
	//RIGA UP: id > imgProp.imageSize - imgProp->width	[7]
	//IN MEZZO											[-1]
	/*
		08 09 10 11
		04 05 06 07
		00 01 02 03
	*/

	int width = imgProp->width;
	int imageSize = imgProp->imageSize;

	// colonna di sinistra
	if (id % width == 0) {
		if (id == 0) //agolo a sinistra
			return 0;
		if (id == imageSize - width) //angolo alto a sx
			return 1;
		return 2; // colonna sinistra
	}
	else if (id % width == width - 1) { // colonna di destra
		if (id == width - 1) //angolo basso a dx
			return 3;
		if (id == imageSize - 1) // angolo alto a destra
			return 4;
		return 5; // colonna a destra
	}
	else if (id < width)
		return 6; // riga in basso
	else if (id > imageSize - width)
		return 7; // riga in altro
	return -1; // pixel centrale
}

__device__ 
void calculateEnergy(energyPixel_t* energyPixel, energyPixel_t* pixel, int id, imgProp_t* imgProp) {

	/// <summary>
	/// Funzione device che calcola l'energia di un pixel.
	/// Con energia si intende l'importanza di quel pixel nell'immagine rispetto ai suoi vicini.
	/// energy = sqrt(dy^2 + dx^2)
	/// dove
	/// dy = pixlexSopra.color - pixelSotto.color
	/// dx = pixlexDestra.color - pixelSinistra.color
	/// Prestando attenzione ai casi di bordo.
	/// </summary>
	/// <param name="energyPixel">L'immagine di output con l'energia calcolata</param>
	/// <param name="pixel">Il pixel di cui si vuole trovare l'energia </param>
	/// <param name="id">L'id del pixel</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine</param>
	
	int dx2, dy2;

	char pos = getPosition(id, imgProp); // ottengo la posizione del pixel (se bordo, se centrale ecc..)
	switch (pos)
	{
	case 0: //angolo in basso a sinistra
		dx2 = energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color;
		break;
	case 1: // alto a destra
		dx2 = energyPixel[id + 1].color;
		dy2 = energyPixel[id - imgProp->width].color;
		break;
	case 2: // colonna a sinistra
		dx2 = energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color - energyPixel[id - imgProp->width].color;
		break;
	case 3: // angolo basso destra
		dx2 = energyPixel[id - 1].color;
		dy2 = energyPixel[id + imgProp->width].color;
		break;
	case 4: //angolo alto a destra
		dx2 = energyPixel[id - 1].color;
		dy2 = energyPixel[id - imgProp->width].color;
		break;
	case 5: //colonna destra
		dx2 = energyPixel[id - 1].color;
		dy2 = energyPixel[id + imgProp->width].color - energyPixel[id - imgProp->width].color;
		break;
	case 6: //bodo sotto
		dx2 = energyPixel[id - 1].color - energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color;
		break;
	case 7: // bordo sopra
		dx2 = energyPixel[id - 1].color - energyPixel[id + 1].color;
		dy2 = energyPixel[id - imgProp->width].color;
		break;
	case -1: // in mezzo
		dx2 = energyPixel[id - 1].color - energyPixel[id + 1].color;
		dy2 = energyPixel[id + imgProp->width].color - energyPixel[id - imgProp->width].color;
		break;
	}

	pixel->energy = sqrtf((dx2 * dx2) + (dy2 * dy2));
}

__device__ 
int min(int id1, int id2, energyPixel_t* energyImg)
{
	/// <summary>
	/// Funzione device che restituisce il minimo tra due pixel basato sulla loro energia
	/// </summary>
	/// <param name="id1">Pixel 1</param>
	/// <param name="id2">Pixel 2</param>
	/// <param name="energyImg">L?immagine con energia</param>
	/// <returns>Il minimo tra id1 e id2</returns>
	
	return (energyImg[id1].energy < energyImg[id2].energy) ? id1 : id2;
}


__global__
void energyMap_(energyPixel_t* energyImg, imgProp_t* imgProp) {

	/// <summary>
	/// Kernel che genera la mappa di energia dell'immagine
	/// </summary>
	/// <param name="energyImg">L'immagine di input di cui calcolare l'energia</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine di input</param>

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imgProp->imageSize) {
		calculateEnergy(energyImg, &energyImg[id], id, imgProp);
	}
}

__global__
void computeMinsPerPixel_(energyPixel_t* energyImg, imgProp_t* imgProp) {
	int idThread = blockIdx.x * blockDim.x + threadIdx.x;


	if (idThread < imgProp->imageSize - imgProp->width) {
		int pos = getPosition(idThread, imgProp);
		switch (pos)
		{
		case 0: // angolo basso a sinistra
		case 2: //colonna a sinistra
			energyImg[idThread].succ_min = min(idThread + imgProp->width, idThread + 1 + imgProp->width, energyImg);

			break;
		case 3: //angolo basso a destra
		case 5: //colonna di destra
			energyImg[idThread].succ_min = min(idThread + imgProp->width, idThread - 1 + imgProp->width, energyImg);

			break;
		case 1: //angolo alto a sinistra
		case 7: // bordo superiore
			break;
		default: //in mezzo o riga inferiore
			energyImg[idThread].succ_min = min(min(idThread + imgProp->width, idThread - 1 + imgProp->width, energyImg),
				idThread + 1 + imgProp->width, energyImg);

			break;
		}

	}

}

__global__ 
void computeSeams_(energyPixel_t* energyImg, pixel_t* imgSrc, seam_t* seams, imgProp_t* imgProp, bool colorSeams = false) {

	/// <summary>
	/// Kernel device che calcola un path dal bordo inferiore a quello superiore. Vengono lanciati N thread pari al numero di pixel di lunghezza
	/// </summary>
	/// <param name="energyImg">L'immagine di input di cui si vogliono trovare i seams</param>
	/// <param name="imgSrc">L'immagine originaria di cui si vuole, eventualmente, colorare i seam trovati.</param>
	/// <param name="seams">Il seam di output trovato</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine trovata</param>
	/// <param name="colorSeams">Se colorare o meno il seam</param>

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;

	int currentId = idThread;
	if (currentId > imgProp->width - 1)
		return;
	int nextIdMin = currentId;


	seams[idThread].total_energy = 0;


	// ogni thread scorre tutta l'immagine in altezza. Avendo lanciato N thread pari alla larghezza, si ha la corretta copertura dell'immagine
	for (int i = 0; i < imgProp->height; i++) { 
		
		currentId = nextIdMin;

		seams[idThread].total_energy += energyImg[currentId].energy;
		seams[idThread].ids[i] = currentId;
		
		//se si vuole colorare il seam
		if (colorSeams) {
			imgSrc[currentId].R = 255;
			imgSrc[currentId].B = 0;
			imgSrc[currentId].G = 0;
		}

		// in base alla posizione verifico quali siano i miei vicini e li considero per trovare il minimo
		int pos = getPosition(currentId, imgProp);
		switch (pos)
		{
		case 0: // angolo basso a sinistra
		case 2: //colonna a sinistra
			nextIdMin = min(currentId + imgProp->width, currentId + 1 + imgProp->width, energyImg);
			break;
		case 3: //angolo basso a destra
		case 5: //colonna di destra
			nextIdMin = min(currentId + imgProp->width, currentId - 1 + imgProp->width, energyImg);
			break;
		case 1: //angolo alto a sinistra
		case 7: // bordo superiore
			break;
		default: //in mezzo o riga inferiore
			nextIdMin =  min(min(currentId + imgProp->width, currentId - 1 + imgProp->width, energyImg),
				currentId + 1 + imgProp->width, energyImg);
			break;
		}

	}
}

__global__
void computeSeams2_(energyPixel_t* energyImg, pixel_t* imgSrc, seam_t* seams, imgProp_t* imgProp, bool colorSeams = false) {

	/// <summary>
	/// Kernel device che calcola un path dal bordo inferiore a quello superiore. Vengono lanciati N thread pari al numero di pixel di lunghezza
	/// </summary>
	/// <param name="energyImg">L'immagine di input di cui si vogliono trovare i seams</param>
	/// <param name="imgSrc">L'immagine originaria di cui si vuole, eventualmente, colorare i seam trovati.</param>
	/// <param name="seams">Il seam di output trovato</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine trovata</param>
	/// <param name="colorSeams">Se colorare o meno il seam</param>

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;

	int currentId = idThread;
	if (currentId > imgProp->width - 1)
		return;

	seams[idThread].total_energy = 0;


	// ogni thread scorre tutta l'immagine in altezza. Avendo lanciato N thread pari alla larghezza, si ha la corretta copertura dell'immagine
	for (int i = 0; i < imgProp->height; i++) {

		seams[idThread].total_energy += energyImg[currentId].energy;
		seams[idThread].ids[i] = currentId;

		currentId = energyImg[currentId].succ_min;

	}
}


__global__
void removeSeam_(energyPixel_t* energyImg, int* idsToRemove, imgProp_t* imgProp, energyPixel_t* newImageGray) {

	/// <summary>
	/// Kernel GPU che rimuove un seam dall'immagine in GS.
	/// Questo kernel viene lanciato con un numero di thread al numero di pixel totali della nuova immagine.
	/// Ogni thread verifica se il pixel che sta considerando si trovi a destra o a sinistra (nella riga) del pixel che deve essere rimosso.
	/// Se a sinistra -> pixel viene copiato nella stessa posizione
	/// Se a destra -> pixel viene shiftato di una posizione a sinistra
	/// Ovviamente viene considerata anche la riga in cui viene fatta, ovvero il numero di pixel eliminati alle righe sottostanti
	/// </summary>
	/// <param name="energyImg">L'immagine di input dalla quale si vuole rimuovere il seam</param>
	/// <param name="idsToRemove">Il path del seam, ovvero tutti gli indici da rimuovere</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine</param>
	/// <param name="newImageGray">La nuova immagine con i pixel rimossi</param>

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;

	if (idThread < imgProp->imageSize) {

		int idRow = idThread / imgProp->width; // la riga di appartenenza del pixel
		int idToRemove = idsToRemove[idRow]; // l'id del pixel da rimuovere
		int shift = idThread < idToRemove ? idRow : idRow + 1; //shift del pixel, 0 se è a sinistra del pixel da rimuovere, 1 se è a destra

		if (idThread == idToRemove)
			return;

		int newPosition = idThread - shift;

		//creazione della nuova immagine con path rimosso
		newImageGray[newPosition].energy = energyImg[idThread].energy;
		newImageGray[newPosition].color = energyImg[idThread].color;
		newImageGray[newPosition].id_pixel = energyImg[idThread].id_pixel;
	}
}

__global__
void updateImageGray_(energyPixel_t* imgGray, energyPixel_t* imgWithoutSeamGray, imgProp_t* imgProp) {

	/// <summary>
	/// Aggiornamento dell'immagine in scala di grigi, con il nuovo path rimosso. Vengono lanciati tanti thread quanti sono i pixel della nuova immagine.
	/// </summary>
	/// <param name="imgGray">La vecchia immagine</param>
	/// <param name="imgWithoutSeamGray">La nuova immagine senza seam</param>
	/// <param name="imgProp">Le caratteristiche della nuova immagine</param>

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgGray[idThread].color = imgWithoutSeamGray[idThread].color;
		imgGray[idThread].id_pixel = imgWithoutSeamGray[idThread].id_pixel;
	}
}

__global__
void updateImageColored_(pixel_t* imgSrc, pixel_t* imgWithoutSeamSrc, imgProp_t* imgProp) {

	/// <summary>
	/// Kernel che aggiorna l'immagine a colori. Vengono lanciati tanti thread quanti sono i pixel dell'immagine senza seam
	/// </summary>
	/// <param name="imgSrc">L'immagine da aggiornare</param>
	/// <param name="imgWithoutSeamSrc">L'immagine con seam rimosso</param>
	/// <param name="imgProp">Le proprietà della nuova immagine</param>

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		imgSrc[idThread].R = imgWithoutSeamSrc[idThread].R;
		imgSrc[idThread].G = imgWithoutSeamSrc[idThread].G;
		imgSrc[idThread].B = imgWithoutSeamSrc[idThread].B;
	}
}

__global__
void removePixelsFromSrc_(pixel_t* imgSrc, pixel_t* newImgSrc, energyPixel_t* imgGray, imgProp_t* imgProp) {

	/// <summary>
	/// Kernel che elimina tutti i pixel dell'immagine a colori basandosi su quelli rimasti nell'immagine in scala di grigi.
	/// Vengono lanciati tanti thread quanti sono i pixel dell'immagine in scala di grigi.
	/// </summary>
	/// <param name="imgSrc">Immagine a colori della quale si vogliono eliminare i pixel</param>
	/// <param name="newImgSrc">L'immagine di output</param>
	/// <param name="imgGray">L'immagine in scala di grigi di input</param>
	/// <param name="imgProp">Le caratteristiche delll'immagine finale</param>

	int idThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (idThread < imgProp->imageSize) {
		newImgSrc[idThread].R = imgSrc[imgGray[idThread].id_pixel].R;
		newImgSrc[idThread].G = imgSrc[imgGray[idThread].id_pixel].G;
		newImgSrc[idThread].B = imgSrc[imgGray[idThread].id_pixel].B;
	}
}


void energyMap(energyPixel_t* energyImg, imgProp_t* imgProp) {
	/// <summary>
	/// Funzione che chiama il kernel GPU che genera la mappa dell'energia
	/// </summary>
	/// <param name="energyImg">L'Immagine di input di cui si vuole calcolare l'energia</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine</param>

	// vengono lanciati tanti kernel quanti sono i pixel nell'immagine
	energyMap_ << <imgProp->imageSize / 1024 + 1, 1024 >> > (energyImg, imgProp);
	gpuErrchk(cudaDeviceSynchronize());
	//writeBMP_energy("src/assets/images/energy.bmp", energyImg, imgProp);
}

void findSeams(energyPixel_t* energyImg, pixel_t* imgSrc, imgProp_t* imgProp, seam_t *minSeam, seam_t* seams, seam_t* minSeamsPerBlock) {
	
	/// <summary>
	/// Funzione host che richiama i kernel computeSeams e min.
	/// Questa funzione permette di trovare il seam da rimuovere mediante la computazione di tutti i seams e della risoluzione di quello minimo.
	/// </summary>
	/// <param name="energyImg">L'immagine di input di cui si vuole trovare il seam da rimuovere</param>
	/// <param name="imgSrc">L'immagine a colori di cui si vogliono colorare i seam</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine di input</param>
	/// <param name="minSeam">Il seam con peso minore da rimuovere</param>
	/// <param name="seams">Lo spazio di memoria dedicato ai seams da trovare</param>
	/// <param name="minSeamsPerBlock">Lo spazio di memoria dedicato a tutti i minseam per ogni blocco del kernel</param>
	
	int nThreads = 128;
	int numBlocksComputeSeams = imgProp->width / nThreads + 1;
	int numBlocksMin = imgProp->width / 1024 + 1;

	//computo tutti i seams
	computeMinsPerPixel_ << <(imgProp->imageSize - imgProp->width) / nThreads + 1, nThreads >> > (energyImg, imgProp);
	//gpuErrchk(cudaDeviceSynchronize());

	computeSeams2_ << <numBlocksComputeSeams, nThreads >> > (energyImg, imgSrc, seams, imgProp);
	//gpuErrchk(cudaDeviceSynchronize());

	//per ogni blocco trovo il seam con peso minore
	min_ <<<numBlocksMin, 1024, 1024 * (sizeof(int) + sizeof(int))>>>(seams, minSeamsPerBlock, imgProp, 1024);
	gpuErrchk(cudaDeviceSynchronize());

	// trovo il seam minore tra quelli dei vari blocchi. Essendo il numero di blocchi molto esiguo, abbiamo preferito eseguirlo lato CPU.
	*minSeam = minSeamsPerBlock[0];
	for (int i = 1; i < numBlocksMin; i++) {
		if (minSeamsPerBlock[i].total_energy < minSeam->total_energy) {
			*minSeam = minSeamsPerBlock[i];
		}
	}

}

void removeSeam(energyPixel_t* imgGray, energyPixel_t* imgWithoutSeamGray, seam_t* idsToRemove, imgProp_t* imgProp) {

	/// <summary>
	/// Funzione host che lancia il kernel GPU che rimuove il seam dall'immagine.
	/// Verranno lanciati tanti thread quanti sono i pixel dell'immagine finale.
	/// Infine aggiorno l'immagine in GS
	/// </summary>
	/// <param name="imgGray"></param>
	/// <param name="imgWithoutSeamGray"></param>
	/// <param name="idsToRemove"></param>
	/// <param name="imgProp"></param>

	// si calcolano le dimensioni dell'immagine finale con seam rimosso
	int newImgSizePixel = imgProp->imageSize - imgProp->height;
	int numBlocks = newImgSizePixel / 1024 + 1;

	// rimuovo il seam
	removeSeam_ << <numBlocks, 1024 >> > (imgGray, idsToRemove->ids, imgProp, imgWithoutSeamGray);
	gpuErrchk(cudaDeviceSynchronize());

	//aggiorno le proprietà dell'immagine
	imgProp->imageSize = newImgSizePixel;
	imgProp->width -= 1;
	
	//aggiorno l'immagine in GS per proseguire con l'iterazione successiva
	updateImageGray_ << <newImgSizePixel / 1024 + 1, 1024 >> > (imgGray, imgWithoutSeamGray, imgProp);
	gpuErrchk(cudaDeviceSynchronize());

}

void removePixelsFromSrc(pixel_t* imgSrc, pixel_t* imgWithoutSeamSrc, energyPixel_t* imgGray, imgProp_t* imgProp) {

	/// <summary>
	/// Funzione host che lancia il kernel GPU utile alla rimozione di tutti i pixel non necessari nell'immagine a colori originale.
	/// Il risultato sarà un immagine ridotta del numero di pixel inseriti dall'utente.
	/// </summary>
	/// <param name="imgSrc">L'immagine originale di input</param>
	/// <param name="imgWithoutSeamSrc">L'immagine finale in output</param>
	/// <param name="imgGray">L'immagine in GS utilizzata per rimuovere i pixel</param>
	/// <param name="imgProp">Le caratteristiche dell'immagine finale.</param>
	
	// Vengono lanciati tanti thread quanti sono i pixel dell'immagine finale da ottenere

	removePixelsFromSrc_ << <imgProp->imageSize / 1024 + 1, 1024 >> > (imgSrc, imgWithoutSeamSrc, imgGray, imgProp);
	gpuErrchk(cudaDeviceSynchronize());
}