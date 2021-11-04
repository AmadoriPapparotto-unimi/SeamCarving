# SeamCarving

L'algoritmo SeamCarving è un algoritmo di ridimensionamento delle immagini content-aware: è in grado di cambiarne dimensione, ma andandone a preservare gli elementi importanti. E' un algoritmo che non scala in dimensione l'immagine, ma va a rimuovere al suo interno i pixel meno importanti.

# Documentazione

Documentazione disponibile [qui](https://amadoripapparotto-unimi.github.io/SeamCarving/) in formato html.

# Esecuzione

L'esecuzione del file `SeamCarving.exe` (che si trova [qui](https://github.com/AmadoriPapparotto-unimi/SeamCarving/releases/tag/v2.0)) prevede due parametri:
- path del file BMP
- numero di iterazioni da applicare

Ad esempio:

```bash
./SeamCarving.exe "C:/path/to/file.bmp" 100
```
  
# Struttura del repository

[src/assets](https://github.com/AmadoriPapparotto-unimi/SeamCarving/tree/main/src/assets): contine immagini utilizzati e i report generati.

[src/bin](https://github.com/AmadoriPapparotto-unimi/SeamCarving/tree/main/src/bin): contiene il codice sorgente.

[report/profiling](https://github.com/AmadoriPapparotto-unimi/SeamCarving/tree/main/src/assets/reports/profiling): alcuni dei test di profiling effettuati.

[report finale](https://github.com/Luca-Tommy/SeamCarving/blob/main/src/assets/reports/Amadori_Papparotto_SeamCarving.pdf).

[SeamCarvingCPU](https://github.com/Luca-Tommy/SeamCarvingCPU): repo github dell'algoritmo SeamCarving implementato a livello di CPU.

[v1.0](https://github.com/AmadoriPapparotto-unimi/SeamCarving/releases/tag/v1.0): soluzione iniziale funzionante ma priva di tutti gli accorgimenti e ottimizzazioni adottate.

[v2.0](https://github.com/AmadoriPapparotto-unimi/SeamCarving/releases/tag/v2.0): soluzione finale.

