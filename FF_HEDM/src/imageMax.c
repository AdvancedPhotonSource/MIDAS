//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void imageMax (char *fn, int Header, int BytesPerPixel, int NrPixelsY, int NrPixelsZ, int nFramesToDo, int startFrameNr){
	FILE *dataFile = fopen(fn,"r");
	fseek(dataFile,Header,SEEK_SET);
	int frameNr;
	char fn2[4096];
	sprintf(fn2,"/dev/shm/img.max");
	FILE *out = fopen(fn2,"w");
	long long int i, j, Pos;
	if (BytesPerPixel == 2){ // GE File
		size_t skip = NrPixelsY*NrPixelsZ*startFrameNr*sizeof(uint16_t);
		fseek(dataFile,skip,SEEK_CUR);
		uint16_t *max;
		max = calloc(NrPixelsY*NrPixelsZ,sizeof(uint16_t));
		uint16_t *data;
		data = malloc(NrPixelsY*NrPixelsZ*sizeof(uint16_t));
		for (frameNr = 0; frameNr < nFramesToDo; frameNr++){
			fread(data,NrPixelsY*NrPixelsZ*sizeof(uint16_t),1,dataFile);
			for (i=0;i<NrPixelsY*NrPixelsZ; i++){
				if (data[i] > max[i]){
					max[i] = data[i];
				}
			}
		}
		fwrite(max,NrPixelsY*NrPixelsZ*sizeof(uint16_t),1,out);
	} else if (BytesPerPixel == 4) {
		size_t skip = NrPixelsY*NrPixelsZ*startFrameNr*sizeof(int32_t);
		fseek(dataFile,skip,SEEK_CUR);
		int32_t *data;
		data = malloc(NrPixelsY*NrPixelsZ*sizeof(int32_t));
		int32_t *max;
		max = calloc(NrPixelsY*NrPixelsZ,sizeof(int32_t));
		for (frameNr = 0; frameNr < nFramesToDo; frameNr++){
			fread(data,NrPixelsY*NrPixelsZ*sizeof(int32_t),1,dataFile);
			for (i=0;i<NrPixelsY*NrPixelsZ; i++){
				if (data[i] > max[i]){
					max[i] = data[i];
				}
			}
		}
		fwrite(max,NrPixelsY*NrPixelsZ*sizeof(int32_t),1,out);
	}
	fclose(out);
}
