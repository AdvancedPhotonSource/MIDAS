//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <nlopt.h>
#include <stdint.h>
#include "nf_headers.h"

#define RealType double
#define float32_t float
#define SetBit(A,k)   (A[(k/32)] |=  (1 << (k%32)))
#define ClearBit(A,k) (A[(k/32)] &= ~(1 << (k%32)))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))

int Flag = 0;
double Wedge;
double Wavelength;
double OmegaRang[MAX_N_OMEGA_RANGES][2];
int nOmeRang;

double**
allocMatrixF(int nrows, int ncols)
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

int**
allocMatrixIntF(int nrows, int ncols)
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

struct my_func_data{
	int NrOfFiles;
    int nLayers;
    double ExcludePoleAngle;
    long long int SizeObsSpots;
    double XGrain[200][3];
    double YGrain[200][3];
    int nSpots;
    double OmegaStart;
    double OmegaStep;
    double px;
    double gs;
    double hkls[5000][4];
    int n_hkls;
    double Thetas[5000];
    int NoOfOmegaRanges;
    int NrPixelsGrid;
    double OmegaRanges[MAX_N_OMEGA_RANGES][2];
    double BoxSizes[MAX_N_OMEGA_RANGES][4];
    double **P0;
    int *ObsSpotsInfo;
};

double IndividualResults[200];

static
double problem_function(
    unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct my_func_data *f_data = (struct my_func_data *) f_data_trial;
	int i, j, count = 1;
	const int NrOfFiles = f_data->NrOfFiles;
    const int nLayers = f_data->nLayers;
    const double ExcludePoleAngle = f_data->ExcludePoleAngle;
    const long long int SizeObsSpots = f_data->SizeObsSpots;
    int nSpots = f_data->nSpots;
    double XGr[nSpots][3];
    double YGr[nSpots][3];
    const double OmegaStart = f_data->OmegaStart;
    const double OmegaStep = f_data->OmegaStep;
    const double px = f_data->px;
    const double gs = f_data->gs;
    const int NoOfOmegaRanges = f_data->NoOfOmegaRanges;
    const int NrPixelsGrid = f_data->NrPixelsGrid;
    double P0[nLayers][3];
    double OmegaRanges[MAX_N_OMEGA_RANGES][2];
    double BoxSizes[MAX_N_OMEGA_RANGES][4];
    double hkls[5000][4];
    int n_hkls = f_data->n_hkls;
    double Thetas[5000];
    for (i=0;i<5000;i++){
		hkls[i][0] = f_data->hkls[i][0];
		hkls[i][1] = f_data->hkls[i][1];
		hkls[i][2] = f_data->hkls[i][2];
		hkls[i][3] = f_data->hkls[i][3];
		Thetas[i] = f_data->Thetas[i];
	}
    int *ObsSpotsInfo;
	ObsSpotsInfo = &(f_data->ObsSpotsInfo[0]);
	for (i=0;i<3;i++){
		for (j=0;j<nSpots;j++){
			XGr[j][i] = f_data->XGrain[j][i];
			YGr[j][i] = f_data->YGrain[j][i];
		}
		for (j=0;j<nLayers;j++){
			P0[j][i] = f_data->P0[j][i];
		}
	}
	for (i=0;i<MAX_N_OMEGA_RANGES;i++){
		for (j=0;j<2;j++){
			OmegaRanges[i][j] = f_data->OmegaRanges[i][j];
		}
		for (j=0;j<4;j++){
			BoxSizes[i][j] = f_data->BoxSizes[i][j];
		}
	}
    double Lsd[nLayers], ybc[nLayers], zbc[nLayers],tx,ty,tz,RotMatTilts[3][3];
    tx = x[0]; ty = x[1]; tz = x[2]; RotationTilts(tx,ty,tz,RotMatTilts);
    Lsd[0] = x[3];
    for (i=1;i<nLayers;i++){
        Lsd[i] = Lsd[i-1] + x[3+i];
    }
    for (i=0;i<nLayers;i++){
        ybc[i] = x[3+nLayers+i];
        zbc[i] = x[3+nLayers+nLayers+i];
    }
	double netResult = 0;
	int ps;
	double *TheorSpots;
	TheorSpots = malloc(MAX_N_SPOTS*3*sizeof(*TheorSpots));
	for (i=0;i<nSpots;i++){
	    double OrientMatIn[3][3], FracOverlap, EulIn[3];
	    //Check euler angles positions!
	    for (j=0;j<3;j++){
			ps = i*3+j+3+(nLayers*3);
			EulIn[j] = x[ps];
		}
	    double XGrain[3],YGrain[3];
	    for (j=0;j<3;j++){
			XGrain[j] = XGr[i][j];
			YGrain[j] = YGr[i][j];
		}
	    Euler2OrientMat(EulIn,OrientMatIn);
	    CalcOverlapAccOrient(NrOfFiles,nLayers,ExcludePoleAngle,Lsd,SizeObsSpots,XGrain,
			YGrain,RotMatTilts,OmegaStart,OmegaStep,px,ybc,zbc,gs,hkls,n_hkls,
			Thetas,OmegaRanges,NoOfOmegaRanges,BoxSizes,P0,NrPixelsGrid,
			ObsSpotsInfo,OrientMatIn,&FracOverlap,TheorSpots);
		netResult += FracOverlap;
		IndividualResults[i] = FracOverlap;
	}
	free(TheorSpots);
	netResult /= nSpots;
    // printf("%.40lf\n",netResult);
    return (1 - netResult);
}

void
FitOrientation(
    const int NrOfFiles,
    const int nLayers,
    const double ExcludePoleAngle,
    double Lsd[nLayers],
    const long long int SizeObsSpots,
    double TiltsOrig[3],
    const double OmegaStart,
    const double OmegaStep,
    const double px,
    double ybc[nLayers],
    double zbc[nLayers],
    const double gs,
    double SpotsInfo[200][9],
    int nSpots,
    double OmegaRanges[MAX_N_OMEGA_RANGES][2],
    const int NoOfOmegaRanges,
    double BoxSizes[MAX_N_OMEGA_RANGES][4],
    double P0[nLayers][3],
    const int NrPixelsGrid,
    int *ObsSpotsInfo,
    double tol,
    double hkls[5000][4],
    double Thetas[5000],
    int n_hkls,
    double **SpotsOut,
    double *LsdFit,
    double *TiltsFit,
    double **BCsFit,
    double tolLsd,
    double tolLsdRel,
    double tolTilts,
    double tolBCsa,
    double tolBCsb)
{
    unsigned n;
    long int i,j;
    n  = 3+(nLayers*3)+(nSpots*3);
    double x[n],xl[n],xu[n];
    for (i=0;i<n;i++){
        x[i] = 0;
        xl[i] = 0;
        xu[i] = 0;
    }
    int count = 0;
    for (i=0;i<3;i++)
    {
        x[i] = TiltsOrig[count];
        xl[i] = x[i] - tolTilts;
        xu[i] = x[i] + tolTilts;
        count++;
    }
    count = 0;
    x[3] = Lsd[0];
    xl[3] = x[3] - tolLsd;
    xu[3] = x[3] + tolLsd;
    count++;
    for (i=4;i<nLayers+3;i++)
    {
        x[i] = Lsd[count] - Lsd[count-1];
        xl[i] = x[i] - tolLsdRel;
        xu[i] = x[i] + tolLsdRel;
        count++;
    }
    count = 0;
    for (i=3+nLayers;i<3+(nLayers*2);i++)
    {
        x[i] = ybc[count];
        x[i+nLayers] = zbc[count];
        xl[i] = x[i] - tolBCsa;
        xl[i+nLayers] = x[i+nLayers] - tolBCsb;
        xu[i] = x[i] + tolBCsa;
        xu[i+nLayers] = x[i+nLayers] + tolBCsb;
        count++;
    }
    count = 0;
    int ps;
    for( i=0; i<nSpots; i++)
    {
		for (j=0;j<3;j++){
			ps = i*3+j+3+(nLayers*3);
			x[ps] = SpotsInfo[i][6+j];
            xl[ps] = x[ps] - (tol*M_PI/180);
			xu[ps] = x[ps] + (tol*M_PI/180);
		}
    }
    //for (i=0;i<n;i++) printf("%f %f %f\n",x[i],xl[i],xu[i]);
	struct my_func_data f_data;
	f_data.NrOfFiles = NrOfFiles;
	f_data.nLayers = nLayers;
	f_data.n_hkls = n_hkls;
	for (i=0;i<5000;i++){
		f_data.hkls[i][0] = hkls[i][0];
		f_data.hkls[i][1] = hkls[i][1];
		f_data.hkls[i][2] = hkls[i][2];
		f_data.hkls[i][3] = hkls[i][3];
		f_data.Thetas[i] = Thetas[i];
	}
	f_data.ExcludePoleAngle = ExcludePoleAngle;
	f_data.SizeObsSpots = SizeObsSpots;
	f_data.P0 = allocMatrixF(nLayers,3);
	for (i=0;i<3;i++){
		for (j=0;j<nSpots;j++){
			f_data.XGrain[j][i] = SpotsInfo[j][2*i];
			f_data.YGrain[j][i] = SpotsInfo[j][2*i+1];
		}
		for (j=0;j<nLayers;j++){
			f_data.P0[j][i] = P0[j][i];
		}
	}
	for (i=0;i<MAX_N_OMEGA_RANGES;i++){
		for (j=0;j<2;j++){
			f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
		}
		for (j=0;j<4;j++){
			f_data.BoxSizes[i][j] = BoxSizes[i][j];
		}
	}
	f_data.ObsSpotsInfo = &ObsSpotsInfo[0];
	f_data.OmegaStart = OmegaStart;
	f_data.OmegaStep = OmegaStep;
	f_data.px = px;
	f_data.gs = gs;
	f_data.nSpots = nSpots;
	f_data.NoOfOmegaRanges = NoOfOmegaRanges;
	f_data.NrPixelsGrid = NrPixelsGrid;
	struct my_func_data *f_datat;
	f_datat = &f_data;
	void* trp = (struct my_func_data *) f_datat;
	double tole = 1e-3;
    
    double val0 = problem_function(n,&x,NULL,trp);
    printf("Original val: %.40lf, running optimization.\n",1-val0);

    nlopt_opt opt = nlopt_create(NLOPT_GN_CRS2_LM,n);
    nlopt_set_population(opt, 500*(n+2));
    nlopt_set_min_objective(opt, problem_function, trp);
    nlopt_set_ftol_rel(opt, 0.001);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	double minf;
	nlopt_optimize(opt, x, &minf);
	nlopt_destroy(opt);
    printf("Refined  val: %.40lf, finished first global optimization. Now doing first local optimization.\n",1-minf);

	nlopt_opt opt2;
	opt2 = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt2, xl);
	nlopt_set_upper_bounds(opt2, xu);
	nlopt_set_min_objective(opt2, problem_function, trp);
	double minf2;
	nlopt_optimize(opt2, x, &minf2);
	nlopt_destroy(opt2);
    printf("Refined  val: %.40lf, finished first local optimization. Now doing second global optimization.\n",1-minf2);

    nlopt_opt opt3 = nlopt_create(NLOPT_GN_ISRES,n);
    nlopt_set_population(opt, 50*(n+2));
    nlopt_set_min_objective(opt3, problem_function, trp);
    nlopt_set_ftol_rel(opt3, 0.01);
	nlopt_set_lower_bounds(opt3, xl);
	nlopt_set_upper_bounds(opt3, xu);
	double minf3;
	nlopt_optimize(opt3, x, &minf3);
	nlopt_destroy(opt3);
    printf("Refined  val: %.40lf, finished second global optimization. Now doing first local optimization.\n",1-minf3);

	nlopt_opt opt4 = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt4, xl);
	nlopt_set_upper_bounds(opt4, xu);
	nlopt_set_min_objective(opt4, problem_function, trp);
	double minf4;
	nlopt_optimize(opt4, x, &minf4);
	nlopt_destroy(opt4);
    printf("Final value:  %.40lf, finished second local optimization. This is the best average confidence.\n",1-minf4);

    TiltsFit[0] = x[0];
    TiltsFit[1] = x[1];
    TiltsFit[2] = x[2];
    LsdFit[0] = x[3];
    for (i=1;i<nLayers;i++){
        LsdFit[i] = LsdFit[i-1] + x[3+i];
    }
    for (i=0;i<nLayers;i++){
        BCsFit[i][0] = x[3+nLayers+i];
        BCsFit[i][1] = x[3+nLayers+nLayers+i];
    }
    for (i=0;i<nSpots;i++){
		for (j=0;j<3;j++){
			ps = i*3+j+3+(nLayers*3);
			SpotsOut[i][j] = x[ps];
		}
		SpotsOut[i][3] = IndividualResults[i];
	}
}

int
main(int argc, char *argv[])
{
    clock_t start, end;
    double diftotal;
    start = clock();
    
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[1000];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[1000];
    int LowNr,nLayers;
    double tx,ty,tz;
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "nDistances ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &nLayers);
            break;
        }
    }
    rewind(fileParam);
    double Lsd[nLayers],ybc[nLayers],zbc[nLayers],ExcludePoleAngle, minFracOverlap,doubledummy;
    double px, OmegaStart,OmegaStep,tol,
           lsdtol,tiltstol,bctola,bctolb,lsdtolrel;
	char fn[1000];
	char fn2[1000];
	char direct[1000];
    double OmegaRanges[MAX_N_OMEGA_RANGES][2], BoxSizes[MAX_N_OMEGA_RANGES][4];
    int cntr=0,countr=0,conter=0,StartNr,EndNr,intdummy,SpaceGroup, RingsToUse[100],nRingsToUse=0;
    int NoOfOmegaRanges=0;
    double SpotsInfo[200][9];
    int nSpots=0;
    Wedge = 0;
    double xc,yc,UD,gSze,gs,eul1,eul2,eul3,ysmall,ybig;
    while (fgets(aline,1000,fileParam)!=NULL){
		str = "ReducedFileName ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, fn2);
            continue;
        }
		str = "DataDirectory ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, direct);
            continue;
        }
		str = "RingsToUse ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingsToUse[nRingsToUse]);
            nRingsToUse++;
            continue;
        }
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Lsd[cntr]);
            cntr++;
            continue;
        }
        str = "SpaceGroup ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &SpaceGroup);
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
        str = "ExcludePoleAngle ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &ExcludePoleAngle);
            continue;
        }
        str = "tx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tx);
            continue;
        }
        str = "ty ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &ty);
            continue;
        }
        str = "BC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &ybc[conter], &zbc[conter]);
            conter++;
            continue;
        }
        str = "tz ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tz);
            continue;
        }
        str = "OrientTol ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tol);
            continue;
        }
        str = "MinFracAccept ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &minFracOverlap);
            continue;
        }
        str = "OmegaStart ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStart);
            continue;
        }
        str = "OmegaStep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStep);
            continue;
        }
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wavelength);
            continue;
        }
        str = "Wedge ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wedge);
            continue;
        }
        str = "px ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &px);
            continue;
        }
        str = "LsdTol ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &lsdtol);
            continue;
        }
        str = "LsdRelativeTol ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &lsdtolrel);
            continue;
        }
        str = "BCTol ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &bctola, &bctolb);
            continue;
        }
        str = "TiltsTol ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tiltstol);
            continue;
        }
        str = "OmegaRange ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &OmegaRanges[NoOfOmegaRanges][0],&OmegaRanges[NoOfOmegaRanges][1]);
            NoOfOmegaRanges++;
            continue;
        }
        str = "BoxSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf", dummy, &BoxSizes[countr][0],
					&BoxSizes[countr][1], &BoxSizes[countr][2], &BoxSizes[countr][3]);
            countr++;
            continue;
        }
        str = "GridSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &gSze);
            continue;
        }
        str = "GridPoints ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s %s %s %lf %lf %s %lf %lf %lf %lf %s %s", dummy, dummy, dummy,
					dummy, &xc, &yc, dummy, &UD, &eul1, &eul2, &eul3, dummy, dummy);
			ysmall  = gSze/(2*sqrt(3));
			ybig  = gSze/(sqrt(3));
			SpotsInfo[nSpots][0] = xc;
			SpotsInfo[nSpots][2] = xc - gSze/2;
			SpotsInfo[nSpots][4] = xc + gSze/2;
			SpotsInfo[nSpots][6] = eul1;
			SpotsInfo[nSpots][7] = eul2;
			SpotsInfo[nSpots][8] = eul3;
			if (UD > 0){ // Upper triangle
				SpotsInfo[nSpots][1] = yc + ybig;
				SpotsInfo[nSpots][3] = yc - ysmall;
				SpotsInfo[nSpots][5] = yc - ysmall;
			}else{ // Lower triangle
				SpotsInfo[nSpots][1] = yc - ybig;
				SpotsInfo[nSpots][3] = yc + ysmall;
				SpotsInfo[nSpots][5] = yc + ysmall;
			}
			nSpots++;
            continue;
        }
        str = "Ice9Input ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            Flag = 1;
            continue;
        }  
    }
    int i,j,m,nrFiles,nrPixels;
    for (i=0;i<NoOfOmegaRanges;i++){
		OmegaRang[i][0] = OmegaRanges[i][0];
		OmegaRang[i][1] = OmegaRanges[i][1];
	}
    nOmeRang = NoOfOmegaRanges;
    gs = gSze;
    fclose(fileParam);
    //Read bin files
    char fnG[1000];
    sprintf(fnG,"%s/grid.txt",direct);
    char fnDS[1000];
    char fnKey[1000];
    char fnOr[1000];
    sprintf(fnDS,"%s/DiffractionSpots.txt",direct);
    sprintf(fnKey,"%s/Key.txt",direct);
    sprintf(fnOr,"%s/OrientMat.txt",direct);
    sprintf(fn,"%s/%s",direct,fn2);
    char *ext="bin";
    int *ObsSpotsInfo;
    int ReadCode;
    nrFiles = EndNr - StartNr + 1;
    nrPixels = 2048*2048;
    long long int SizeObsSpots, iT;
    SizeObsSpots = (nLayers);
    SizeObsSpots*=nrPixels;
    SizeObsSpots*=nrFiles;
    SizeObsSpots/=32;
    ObsSpotsInfo = malloc(SizeObsSpots*sizeof(*ObsSpotsInfo));
    for (iT=0;iT<SizeObsSpots;iT++){
		ObsSpotsInfo[i] = 0;
	}
    memset(ObsSpotsInfo,0,SizeObsSpots*sizeof(*ObsSpotsInfo));
    printf("Size of spot info: %llu mb\n",SizeObsSpots*sizeof(int)/(1024*1024));
    if (ObsSpotsInfo==NULL){
        printf("Could not allocate ObsSpotsInfo.\n");
        return 0;
    }
    ReadCode = ReadBinFiles(fn,ext,StartNr,EndNr,ObsSpotsInfo,nLayers,SizeObsSpots);
    if (ReadCode == 0){
        printf("Reading bin files did not go well. Please check.\n");
        return 0;
    }
	double *LsdFit, *TiltsFit, **BCsFit;
	double TiltsOrig[3];
	TiltsOrig[0] = tx;
	TiltsOrig[1] = ty;
	TiltsOrig[2] = tz;
	LsdFit = malloc(nLayers*sizeof(*LsdFit));
	TiltsFit = malloc(nLayers+sizeof(*TiltsFit));
	BCsFit = allocMatrixF(nLayers,2);
	int n_hkls = 0;
	double hkls[5000][4];
	double Thetas[5000];
	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	fgets(aline,1000,hklf);
    double RotMatTilts[3][3];
    RotationTilts(tx,ty,tz,RotMatTilts);
	double MatIn[3],P0[nLayers][3],P0T[3];
	double **SpotsOut;
	SpotsOut = allocMatrixF(nSpots,4);
    MatIn[0]=0;
    MatIn[1]=0;
    MatIn[2]=0;
    for (i=0;i<nLayers;i++){
        MatIn[0] = -Lsd[i];
        MatrixMultF(RotMatTilts,MatIn,P0T);
        for (j=0;j<3;j++){
            P0[i][j] = P0T[j];
        }
    }
    int NrPixelsGrid=2*(ceil((gs*2)/px))*(ceil((gs*2)/px));
    while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %lf %lf %lf %lf %lf %s %s",dummy,dummy,dummy,
			dummy,&hkls[n_hkls][3],&hkls[n_hkls][0],&hkls[n_hkls][1],
			&hkls[n_hkls][2],&Thetas[n_hkls],dummy,dummy);
		n_hkls++;
	}
	if (nRingsToUse > 0){
		double hklTemps[n_hkls][4],thetaTemps[n_hkls];
		int totalHKLs=0;
		for (i=0;i<nRingsToUse;i++){
			for (j=0;j<n_hkls;j++){
				if ((int)hkls[j][3] == RingsToUse[i]){
					hklTemps[totalHKLs][0] = hkls[j][0];
					hklTemps[totalHKLs][1] = hkls[j][1];
					hklTemps[totalHKLs][2] = hkls[j][2];
					hklTemps[totalHKLs][3] = hkls[j][3];
					thetaTemps[totalHKLs] = Thetas[j];
					totalHKLs++;
				}
			}
		}
		for (i=0;i<totalHKLs;i++){
			hkls[i][0] = hklTemps[i][0];
			hkls[i][1] = hklTemps[i][1];
			hkls[i][2] = hklTemps[i][2];
			hkls[i][3] = hklTemps[i][3];
			Thetas[i] = thetaTemps[i];
		}
		n_hkls = totalHKLs;
	}
	printf("Number of individual diffracting planes: %d\n",n_hkls);
	FitOrientation(nrFiles,nLayers,ExcludePoleAngle,Lsd,SizeObsSpots,
			TiltsOrig,OmegaStart,OmegaStep,px,ybc,zbc,gs,SpotsInfo,nSpots,
			OmegaRanges,NoOfOmegaRanges,BoxSizes,P0,NrPixelsGrid,
			ObsSpotsInfo,tol,hkls,Thetas,n_hkls,SpotsOut,LsdFit,TiltsFit,BCsFit,lsdtol,
			lsdtolrel,tiltstol,bctola,bctolb);
	for (i=0;i<nLayers;i++){
		printf("Lsd %f\n",LsdFit[i]);
	}
	for (i=0;i<nLayers;i++){
		printf("BC %f %f\n",BCsFit[i][0],BCsFit[i][1]);
	}
	printf("tx %f\nty %f\ntz %f\n",TiltsFit[0],TiltsFit[1],TiltsFit[2]);
	printf("EulerAngle1 EulerAngle2 EulerAngle3 Confidence\n");
	for (i=0;i<nSpots;i++){
		for (j=0;j<4;j++){
			printf("%f ",SpotsOut[i][j]);
		}
		printf("\n");
	}
    end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed in comparing diffraction spots: %f [s]\n",diftotal);
    return 0;
}
