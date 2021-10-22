# Note aggiuntive sullo sviluppo

## Conversione a scala di grigi e creazione mappa energia

* IDEA 1: 
	- Suddividere immagine in sotto-immagini
	- Creare uno stream per ogni sotto-immagine e (successivamente) calcolare scala di grigi e successivamente energia
	- L'esecuzione degli stream avviene in parallelo
	- Il calcolo dei bordi avviene alla fine dell'esecuzione di tutti gli stream

## Meglio parallelismo dinamico o singolo kernel? (seam_carving.cu:removeSeam)

