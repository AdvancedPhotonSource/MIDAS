//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  MergeOverlappingPeaks.c
//
//
//  Created by Hemant Sharma on 2014/07/14.
//
//
//

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <sys/types.h>
#include <errno.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MAXNHKLS 5000
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y)   sqrt((x)*(x) + (y)*(y))
#define nOverlapsMaxPerImage 10000

static inline 
double CalcEtaAngle(double y, double z){
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
	return alpha;
}

static inline
void YZ4mREta(int NrElements, double *R, double *Eta, double *Y, double *Z){
	int i;
	for (i=0;i<NrElements;i++){
		Y[i] = -R[i]*sin(Eta[i]*deg2rad);
		Z[i] = R[i]*cos(Eta[i]*deg2rad);
	}
}

static inline
int**
allocMatrixInt(int nrows, int ncols)
{
    int** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline
void
FreeMemMatrixInt(int **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline
double**
allocMatrix(int nrows, int ncols)
{
    double** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline
void
FreeMemMatrix(double **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

struct InputData{
	double SpotID;
	double IntegratedIntensity;
	double Omega;
	double YCen;
	double ZCen;
	double IMax;
	double Radius;
	double Eta;
	double SigmaR;
	double SigmaEta;
};

static int cmpfunc (const void * a, const void *b){
	struct InputData *ia = (struct InputData *)a;
	struct InputData *ib = (struct InputData *)b;
	return (int)(1000.f*ia->Eta - 1000.f*ib->Eta);
}

static inline int CheckDirectoryCreation(char Folder[1024],char FileStem[1024])
{
	int e;
    struct stat sb;
	char totOutDir[1024];
	sprintf(totOutDir,"%s/PeakSearch/",Folder);
    e = stat(totOutDir,&sb);
    if (e!=0 && errno == ENOENT){
		printf("Output directory did not exist, creating %s\n",totOutDir);
		e = mkdir(totOutDir,S_IRWXU);
		if (e !=0) {printf("Could not make the directory. Exiting\n");return 0;}
	}
	sprintf(totOutDir,"%s/PeakSearch/%s",Folder,FileStem);
    e = stat(totOutDir,&sb);
    if (e!=0 && errno == ENOENT){
		printf("Output directory did not exist, creating %s\n",totOutDir);
		e = mkdir(totOutDir,S_IRWXU);
		if (e !=0) {printf("Could not make the directory. Exiting\n");return 0;}
	}
	return 1;
}

static inline int ReadSortFiles (char OutFolderName[1024], char FileStem[1024], int FileNr, int RingNr, int Padding, double **SortedMatrix)
{
	char aline[1000];
    char InFile[1024];
    sprintf(InFile,"%s/%s_PS.csv",OutFolderName,FileStem);
    FILE *infileread;
    infileread = fopen(InFile,"r");
    struct InputData *MyData;
    MyData = malloc(nOverlapsMaxPerImage*sizeof(*MyData));
    int counter = 0;
    fgets(aline,1000,infileread);
    double SpotID,IntegratedIntensity,Omega,YCen,ZCen,IMax,Radius,Eta,NumberOfPixels;
    int RingNrT;
    char dummy[2048];
    while (fgets(aline,1000,infileread)!=NULL){
		sscanf(aline,"%s %s %s %s %s %s %s %s %s %s %d", dummy, dummy, dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,&RingNrT);
		if (RingNrT != RingNr) continue;
		sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
					&(MyData[counter].SpotID), &(MyData[counter].IntegratedIntensity), &(MyData[counter].Omega),
					&(MyData[counter].YCen), &(MyData[counter].ZCen), &(MyData[counter].IMax), &(MyData[counter].Radius),
					&(MyData[counter].Eta), &(MyData[counter].SigmaR), &(MyData[counter].SigmaEta));
		counter++;
	}
	fclose(infileread);
    qsort(MyData, counter, sizeof(struct InputData), cmpfunc);
    int i,j,counter2=0;
    for (i=0;i<counter;i++){
		if (MyData[i].IntegratedIntensity < 1){
			continue;
		}
		SortedMatrix[counter2][0] = MyData[i].SpotID;
		SortedMatrix[counter2][1] = MyData[i].IntegratedIntensity;
		SortedMatrix[counter2][2] = MyData[i].Omega;
		SortedMatrix[counter2][3] = MyData[i].YCen;
		SortedMatrix[counter2][4] = MyData[i].ZCen;
		SortedMatrix[counter2][5] = MyData[i].IMax;
		SortedMatrix[counter2][6] = MyData[i].Radius;
		SortedMatrix[counter2][7] = MyData[i].Eta;
		SortedMatrix[counter2][8] = MyData[i].SigmaR;
		SortedMatrix[counter2][9] = MyData[i].SigmaEta;
		counter2++;
	}
	free(MyData);
    return counter2;
}

int main(int argc, char *argv[]){
	clock_t start, end;
    double diftotal;
    start = clock();
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    int RingNr = atoi(argv[2]);
    char aline[1000], *str, dummy[1000];
    fileParam = fopen(ParamFN,"r");
    int LowNr = 1;
    char Folder[1024], FileStem[1024],*TmpFolder,fs[1024];
    int LayerNr;
    int StartNr, EndNr, Padding=6;
    TmpFolder = "Temp";
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "Folder ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, Folder);
            continue;
        }
        str = "FileStem ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, fs);
            continue;
        }
        str = "Padding ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &Padding);
            continue;
        }
        str = "LayerNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &LayerNr);
            continue;
        }
        str = "StartNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &StartNr);
            continue;
        }
        str = "EndNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &EndNr);
            continue;
        }
	}
	sprintf(FileStem,"%s_%d",fs,LayerNr);
	fclose(fileParam);
	double MarginOmegaOverlap = sqrt(4);
	int TotNrFiles = EndNr - StartNr + 1, i,j,k;
	int nFilesMax = 50;
    char OutFolderName[1024];
    char OutFileName[1024];
    sprintf(OutFolderName,"%s/%s",Folder,TmpFolder);
    char header[1024] = "SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px)"
					" IMax MinOme(degrees) MaxOme(degress) SigmaR SigmaEta\n";
    
    // Read first file
    int FileNr = StartNr;
	int nSpots,nSpotsNew;
	double **NewIDs, **CurrentIDs, **TempIDs;
	NewIDs = allocMatrix(nOverlapsMaxPerImage,10);
	CurrentIDs = allocMatrix(nOverlapsMaxPerImage,14);
	TempIDs = allocMatrix(nOverlapsMaxPerImage,14);
    nSpots = ReadSortFiles(OutFolderName,FileStem,FileNr,RingNr,Padding,NewIDs);
    for (i=0;i<nSpots;i++){
		CurrentIDs[i][0] = NewIDs[i][0];              // SpotID
		CurrentIDs[i][1] = NewIDs[i][1];              // IntegratedIntensity
		CurrentIDs[i][2] = NewIDs[i][2]*NewIDs[i][1]; // Omega*IntegratedIntensity
		CurrentIDs[i][3] = NewIDs[i][3]*NewIDs[i][1]; // YCen*IntegratedIntensity
		CurrentIDs[i][4] = NewIDs[i][4]*NewIDs[i][1]; // ZCen*IntegratedIntensity
		CurrentIDs[i][5] = NewIDs[i][5];              // IMax
		CurrentIDs[i][6] = NewIDs[i][6];              // Radius
		CurrentIDs[i][7] = NewIDs[i][7];              // Eta
		CurrentIDs[i][8] = NewIDs[i][3];			  // YCen
		CurrentIDs[i][9] = NewIDs[i][4];			  // ZCen
		CurrentIDs[i][10] = NewIDs[i][2];			  // MinOmega
		CurrentIDs[i][11] = NewIDs[i][2];			  // MaxOmega
		CurrentIDs[i][12] = NewIDs[i][8];			  // SigmaR
		CurrentIDs[i][13] = NewIDs[i][9];			  // SigmaEta
	}
    int e = CheckDirectoryCreation(Folder,FileStem);
    if (e ==0) return 1;
    sprintf(OutFileName,"%s/PeakSearch/%s/Result_StartNr_%d_EndNr_%d_RingNr_%d.csv",Folder,FileStem,StartNr,EndNr,RingNr);
	FILE *OutFile;
	OutFile = fopen(OutFileName,"w");
	fprintf(OutFile,"%s",header);
	double diffLen,yThis,zThis,minLen,yFwd,zFwd,diffLenFwd;
	int *TempIDsCurrent,*TempIDsNew,BestID,IDFound;
	TempIDsCurrent = malloc(nOverlapsMaxPerImage*sizeof(*TempIDsCurrent));
	TempIDsNew = malloc(nOverlapsMaxPerImage*sizeof(*TempIDsNew));
	memset(TempIDsCurrent,0,nOverlapsMaxPerImage*sizeof(*TempIDsCurrent));
	memset(TempIDsNew,0,nOverlapsMaxPerImage*sizeof(*TempIDsNew));
	int SpotIDNr = 1,counter;
    if (StartNr==EndNr){ // If there is only one file.
		for (i=0;i<nSpots;i++){
			fprintf(OutFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",(int)NewIDs[i][0],NewIDs[i][1],NewIDs[i][2],
												  NewIDs[i][3],NewIDs[i][4],NewIDs[i][5],NewIDs[i][2],NewIDs[i][2],
												  NewIDs[i][8],NewIDs[i][9]);
		}
	}else{ // If there are multiple files:
		for (FileNr=(StartNr+1);FileNr<=EndNr;FileNr++){
			nSpotsNew = ReadSortFiles(OutFolderName,FileStem,FileNr,RingNr,Padding,NewIDs);
			//printf("%d\n",FileNr);
			for (i=0;i<nSpots;i++){
				minLen = 10000000;
				IDFound = 0;
				yThis = CurrentIDs[i][8];
				zThis = CurrentIDs[i][9];
				for (j=0;j<nSpotsNew;j++){ // Try to find the smallest difference in Y,Z.
					if (TempIDsNew[j]!=1){
						diffLen = CalcNorm2(NewIDs[j][3]-yThis,NewIDs[j][4]-zThis);
						if (diffLen < MarginOmegaOverlap && diffLen<minLen){
							minLen = diffLen;
							BestID = j;
							IDFound = 1;
						}
					}
				}
				if (IDFound == 1){ // If a candidate for overlapping has been detected, check if it is the best pair.
					yFwd = NewIDs[BestID][3];
					zFwd = NewIDs[BestID][4];
					for (k=0;k<nSpots;k++){
						if (k!=i && TempIDsCurrent[k]!=1){
							diffLenFwd = CalcNorm2(CurrentIDs[k][8]-yFwd,CurrentIDs[k][9]-zFwd);
							if (diffLenFwd < minLen){
								IDFound = 0;
								break;
							}
						}
					}
				}
				if (IDFound == 1){ // If the best pair for overlapping was found, update current IDs.
					TempIDsCurrent[i] = 1;
					TempIDsNew[BestID] = 1;
					CurrentIDs[i][1] +=  NewIDs[BestID][1];
					CurrentIDs[i][2] += (NewIDs[BestID][2]*NewIDs[BestID][1]);
					CurrentIDs[i][3] += (NewIDs[BestID][3]*NewIDs[BestID][1]);
					CurrentIDs[i][4] += (NewIDs[BestID][4]*NewIDs[BestID][1]);
					if (CurrentIDs[i][5] < NewIDs[BestID][5]){
						CurrentIDs[i][5] =  NewIDs[BestID][5]; // IMax update
					}
					CurrentIDs[i][8] =  NewIDs[BestID][3]; // Ycen
					CurrentIDs[i][9] =  NewIDs[BestID][4]; // ZCen
					if (CurrentIDs[i][10] > NewIDs[BestID][2]){
						CurrentIDs[i][10] =  NewIDs[BestID][2]; // MinOme
					}
					if (CurrentIDs[i][11] < NewIDs[BestID][2]){
						CurrentIDs[i][11] =  NewIDs[BestID][2]; // MaxOme
					}
					if (CurrentIDs[i][12] < NewIDs[BestID][8]){
						CurrentIDs[i][12] =  NewIDs[BestID][8]; // SigmaR
					}
					if (CurrentIDs[i][13] < NewIDs[BestID][9]){
						CurrentIDs[i][13] =  NewIDs[BestID][9]; // SigmaEta
					}
				}
			}
			//Write all the spots not overlapping to the output file.
			for (i=0;i<nSpots;i++){
				if (TempIDsCurrent[i] == 0){ // Spot was not overlapping.
					fprintf(OutFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",SpotIDNr,
							CurrentIDs[i][1],(CurrentIDs[i][2]/CurrentIDs[i][1]),
							(CurrentIDs[i][3]/CurrentIDs[i][1]),(CurrentIDs[i][4]/CurrentIDs[i][1]),
							CurrentIDs[i][5],CurrentIDs[i][10],CurrentIDs[i][11],CurrentIDs[i][12],
							CurrentIDs[i][13]);
					SpotIDNr++;
				}
			}
			// Reset everything for the next file.
			counter = 0;
			for (i=0;i<nSpots;i++){
				if (TempIDsCurrent[i] == 1){ // Spot was overlapping.
					for (j=0;j<12;j++){
						TempIDs[counter][j] = CurrentIDs[i][j];
					}
					counter++;
				}
			}
			for (i=0;i<nSpotsNew;i++){
				if (TempIDsNew[i] == 0){ // Spot was not overlapping.
					TempIDs[counter][0] = NewIDs[i][0];              // SpotID
					TempIDs[counter][1] = NewIDs[i][1];              // IntegratedIntensity
					TempIDs[counter][2] = NewIDs[i][2]*NewIDs[i][1]; // Omega*IntegratedIntensity
					TempIDs[counter][3] = NewIDs[i][3]*NewIDs[i][1]; // YCen*IntegratedIntensity
					TempIDs[counter][4] = NewIDs[i][4]*NewIDs[i][1]; // ZCen*IntegratedIntensity
					TempIDs[counter][5] = NewIDs[i][5];              // IMax
					TempIDs[counter][6] = NewIDs[i][6];              // Radius
					TempIDs[counter][7] = NewIDs[i][7];              // Eta
					TempIDs[counter][8] = NewIDs[i][3];			     // YCen
					TempIDs[counter][9] = NewIDs[i][4];		  	     // ZCen
					TempIDs[counter][10] = NewIDs[i][2];			 // MinOmega
					TempIDs[counter][11] = NewIDs[i][2];			 // MaxOmega
					TempIDs[counter][12] = NewIDs[i][8];			 // SigmaR
					TempIDs[counter][13] = NewIDs[i][9];			 // SigmaEta
					counter++;
				}
			}
			if (counter != nSpotsNew){
				printf("Number of spots mismatch. Please have a look.\n");
			}
			for (i=0;i<nSpots;i++){
				for (j=0;j<12;j++){
					CurrentIDs[i][j] = 0;
				}
			}
			for (i=0;i<nSpotsNew;i++){
				for (j=0;j<12;j++){
					CurrentIDs[i][j] = TempIDs[i][j];
				}
			}
			nSpots = nSpotsNew;
			memset(TempIDsCurrent,0,nOverlapsMaxPerImage*sizeof(*TempIDsCurrent));
			memset(TempIDsNew,0,nOverlapsMaxPerImage*sizeof(*TempIDsNew));
		}
	}
	for (i=0;i<nSpots;i++){
		fprintf(OutFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",SpotIDNr,
				CurrentIDs[i][1],(CurrentIDs[i][2]/CurrentIDs[i][1]),
				(CurrentIDs[i][3]/CurrentIDs[i][1]),(CurrentIDs[i][4]/CurrentIDs[i][1]),
				CurrentIDs[i][5],CurrentIDs[i][10],CurrentIDs[i][11],CurrentIDs[i][12],
				CurrentIDs[i][13]);
		SpotIDNr++;
	}
	FreeMemMatrix(NewIDs,nOverlapsMaxPerImage);
	FreeMemMatrix(CurrentIDs,nOverlapsMaxPerImage);
	FreeMemMatrix(TempIDs,nOverlapsMaxPerImage);
	free(TempIDsCurrent);
	free(TempIDsNew);
    end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
