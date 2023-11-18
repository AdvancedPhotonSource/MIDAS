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
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>
#include <unistd.h>
#include "tomo_heads.h"
#include <sys/sysinfo.h>

/*
 * The data can be one of two types:
 * 							sinogram already with float data type, directly give to reconstruct code with some additional centering etc.
 * 							dark, whites (2) and then raw images. Using number of angles, we know how many images are there. The scaling with white should be proportional to the distance from a white and appropriate dark value.
*/

// TODO:
//	1. Check size of arrays needed and then allocate number of threads accordingly.
//	2. If nSlices*nShifts is not a multiple of nThreads, correctly calculate the number of
//	3. Safe malloc to check NULL ptrs.

void usage(){
	printf("MIDAS-TOMO Code to do tomo recon using Gridrec. Based on tomompi implementation from Brian Tiemann, APS. Maintained by Hemant Sharma, APS (hsharma@anl.gov).\nUsage is: \n"
		"tomo ParamsFile.txt numberOfParallelJobs\n"
		"Params file must have the following parameters:\n"
		"Input file is a text file name with a data link: sino data is a !!!single!!! binary file with darks, whites and tomo data in that order.\n"
		"* The rest of the file consists of the parameters required.\n"
		"* Parameters to be supplied:\n"
		"	* saveReconSeparate (optional): [int] 0 if want to save in recon in single file, 1 if want to save in individual files.\n"
		"	* dataFileName: [char*] name of the file with the raw data or sino data\n"
		"	* reconFileName: [char*] Name of the file for saving the reconstruction\n"
		"	* areSinos: If the input is a sinogram instead of raw (cleaned) images [0 or 1]\n"
		"	* The data can be one of two types: \n"
		"	* 							sinogram already with float data type, directly give to reconstruct code with some additional centering etc. \n"
		"	* 							dark[float], whites (2,floats) and then raw images[shorts]. Using number of angles, we know how many images are there.\n"
		"	*							The scaling with white should be proportional to the distance from a white and appropriate dark value.\n"
		"	* detXdim - [uint]\n"
		"	* detYdim - [uint]\n"
		"	* Thetas can either be given as a range:\n"
		"	* 	thetaRange: startAngle endAngle angleInterval - [floats]\n"
		"	* or a File:\n"
		"	* 	thetaFileName [char*] with each line having an angle value [float].\n"
		"	* filter - [int] set to * 0: default\n"
		"							* 1: Shepp / Logan\n"
		"							* 2: Hann\n"
		"							* 3: Hamming\n"
		"							* 4: Ramp\n"
		"	* shiftValues: start_shift end_shift shift_interval [floats] In case of 1 shift, give start_shift=end_shift, shift_interval doesn't matter.\n"
		"	*					ENSURE TO GIVE A RANGE WITH EVEN NUMBER OF SHIFTS\n"
		"	* ringRemovalCoefficient - If given, will do ringRemoval, otherwise comment or remove line [float] default 1.0\n"
		"   * doLog - If 1, will take Log of intensities to calculate transmission, otherwise will use intensities directly. [int] default 1.\n"
		"	* slicesToProcess - -1 for all or FileName. ENSURE TO GIVE EVEN NUMBER OF SLICES\n"
		"	* ExtraPad - 0 if half padding, 1 if one-half padding"
		"	* AutoCentering - 0 if don't want reconstruction shifted in one direction (rotation axis in center of recon)\n"
		"	* 				- 1 if want shift (rotation axis is offset) [default]\n"
		"Output file: float with reconstruction_xdim*reconstruction_xdim size\n"
		"OutputFileName: {recon_info_record.ReconFileName}_sliceNr_reconstruction_xdim_reconstruction_xdim_float_4byte.bin\n"
		"The code will generate two text files: fftwf_wisdom_{1,2}d.txt. "
		"These files are ways to speed up the fft calculation.\n"
		"First run on a dataset generates these files which can be used to speed up subsequent runs.\n");
}

int main(int argc, char *argv[])
{
	if (argc!=3){
		usage();
		return 1;
	}
	GLOBAL_CONFIG_OPTS recon_info_record;
	recon_info_record.sizeMatrices = 0;
	char *fileName;
	fileName = argv[1];
	int RC;
	RC = setGlobalOpts(fileName, &recon_info_record);
	setReadStructSize(&recon_info_record);
	gridrecParams pm;
	pm.sinogram_x_dim = recon_info_record.sinogram_adjusted_xdim * 2;
	getGridRecFourSizes(&pm);
	int fftw1d_size = (int)pm.pdim;
	int fftw2d_size = (int)pm.M;
	char plan2DFN[4096];
	sprintf(plan2DFN,"fftwf_wisdom_2d_%d.txt",(int)fftw2d_size);
	char plan1DFN[4096];
	sprintf(plan1DFN,"fftwf_wisdom_1d_%d.txt",(int)fftw1d_size);
	if (RC!=0){
		printf("Parameter file could not be read. Exiting.\n");
		return 1;
	}
	// Get FFT Plan
	if (access (plan2DFN, F_OK) == -1){
		printf("FFT plan file did not exist, creating one %s.\n",plan2DFN);		// Check if sizes are okay.
		createPlanFile(&recon_info_record);
	} else if(access (plan1DFN, F_OK) == -1) {
		printf("FFT plan file did not exist, creating one %s.\n",plan2DFN);
		createPlanFile(&recon_info_record);
	} else {
		printf("Reading wisdom file %s.\n",plan2DFN);
		createPlanFile(&recon_info_record);
	}
	// Read /proc/meminfo to get the available RAM size and calculate maxNProcs accordingly.
	long long int avRAM;
	FILE *memf = fopen("/proc/meminfo","r");
	char aline[4096], dummy[4096];
	while (fgets(aline,4096,memf)!= NULL){
		if (strncmp(aline,"MemAvailable:",strlen("MemAvailable:"))==0){
			sscanf(aline,"%s %lld",dummy,&avRAM);
			break;
		}
	}
	avRAM *= 1024; // Get in bytes
	recon_info_record.sizeMatrices*= 2;
	long long int maxNProcs = (long long int) avRAM / (long long int) recon_info_record.sizeMatrices;
	int numProcs = (atoi(argv[2]) < maxNProcs-2) ? atoi(argv[2]) : maxNProcs - 2;
	printf("Memory needed per process: %lld, Total available RAM: %lld, MaxNProcs: %lld.\nWe can run up to %lld processes.\nWe will use %lld MB RAM\n",
			(long long int) recon_info_record.sizeMatrices,avRAM, maxNProcs, maxNProcs-2, (long long int)numProcs*recon_info_record.sizeMatrices/(1024*1024*1024));
	// Check if sizes are okay.
	if (recon_info_record.n_shifts > 1 && recon_info_record.n_shifts %2 !=0){
		printf("Number of shifts must be even. Exiting\n");
		return 1;
	}
	if (recon_info_record.n_shifts == 1 && recon_info_record.n_slices %2 !=0){
		printf("Number of slices must be even. Exiting\n");
		return 1;
	}
	int rc = fftwf_import_wisdom_from_filename("fftwf_wisdom_1d.txt");
	double start_time = omp_get_wtime();
	if (recon_info_record.n_shifts==1){
		printf("Starting processing of all slices with %d threads.\n",numProcs);
		# pragma omp parallel num_threads(numProcs)
		{
			int procNr = omp_get_thread_num();
			int nrSlicesThread = (int)ceil((double)recon_info_record.n_slices / (2.0*(double)numProcs));
			int startSliceNr = procNr*nrSlicesThread*2;
			int endSliceNr = startSliceNr + nrSlicesThread*2;
			if (endSliceNr > recon_info_record.n_slices) endSliceNr = recon_info_record.n_slices;
			// Allocate all the structs and arrays now
			SINO_READ_OPTS readStruct;
			readStruct.norm_sino = (float *) malloc(sizeof(float)*recon_info_record.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
			LOCAL_CONFIG_OPTS information;
			information.shift = recon_info_record.shift_values[0];
			setSinoSize(&information,recon_info_record);
			gridrecParams param;
			param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2;
			param.theta_list = recon_info_record.theta_list;
			param.filter_type = recon_info_record.filter;
			param.theta_list_size = recon_info_record.theta_list_size;
			param.wisdom_string = (char *) malloc(sizeof(char) * (strlen(recon_info_record.wisdom_string)+1));
			param.setPlan = 0;
			strcpy(param.wisdom_string,recon_info_record.wisdom_string);
			size_t offt, offsetRecons;
			setGridRecPSWF(&param);
			initFFTMemoryStructures(&param);
			initGridRec(&param);
			int numSlice, sliceRowNr, oldSliceNr;
			for (numSlice = 0; numSlice<(endSliceNr-startSliceNr)/2; numSlice++){
				memset(readStruct.norm_sino,0,sizeof(float)*recon_info_record.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				memsets(&information,recon_info_record);
				int sliceNr;
				sliceRowNr = startSliceNr + numSlice*2;
				sliceNr = recon_info_record.slices_to_process[sliceRowNr];
				oldSliceNr = sliceNr;
				if (recon_info_record.are_sinos){
					int rc = readSino(sliceNr,recon_info_record,&readStruct);
					if (rc == 1) continue;
				} else {
					int rc = readRaw(sliceNr,recon_info_record,&readStruct);
					if (rc == 1) continue;
				}
				memcpy(information.sino_calc_buffer,readStruct.norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				offt = 0;
				offsetRecons = 0;
				reconCentering(&information,recon_info_record,offt,recon_info_record.doLogProj);
				setSinoAndReconBuffers(1, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				sliceRowNr ++;
				sliceNr = recon_info_record.slices_to_process[sliceRowNr];
				if (recon_info_record.are_sinos){
					int rc = readSino(sliceNr,recon_info_record,&readStruct);
					if (rc == 1) continue;
				} else {
					int rc = readRaw(sliceNr,recon_info_record,&readStruct);
					if (rc == 1) continue;
				}
				memcpy(information.sino_calc_buffer,readStruct.norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				offt = information.sinogram_adjusted_size*2;
				offsetRecons = information.reconstruction_size*4;
				reconCentering(&information,recon_info_record,offt,recon_info_record.doLogProj);
				setSinoAndReconBuffers(2, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				reconstruct(&param);
				getRecons(&information,recon_info_record,&param,0);
				int rw = writeRecon(oldSliceNr,&information,recon_info_record,0);
				if (rw == 1) continue;
				getRecons(&information,recon_info_record,&param,offsetRecons);
				rw = writeRecon(sliceNr,&information,recon_info_record,0);
				if (rw == 1) continue;
			}
			destroyFFTMemoryStructures(&param);
		}
	} else { // We have multiple shifts, (possibly multiple slices_to_process)
		SINO_READ_OPTS *readStruct;
		readStruct = malloc(recon_info_record.n_slices*sizeof(*readStruct));
		int i;
		for (i = 0; i < recon_info_record.n_slices; i ++)
			readStruct[i].norm_sino = (float *) malloc(sizeof(float)*recon_info_record.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
		// ReadStruct is now ready.
		//~ int nJobs = (numProcs < recon_info_record.n_slices) ? numProcs : recon_info_record.n_slices;
		int nJobs = recon_info_record.n_slices;
		int badRead = 0;
		# pragma omp parallel num_threads(nJobs)
		{
			// This will only read up to nJobs slices. TODO: Fix this!!!
			int procNr = omp_get_thread_num();
			int sliceNr;
			sliceNr = recon_info_record.slices_to_process[procNr];
			//~ printf("Reading SliceNr: %d.\n",sliceNr);
			if (recon_info_record.are_sinos){
				int rc = readSino(sliceNr,recon_info_record,&readStruct[procNr]);
				if (rc == 1) badRead = 1;
			} else {
				int rc = readRaw(sliceNr,recon_info_record,&readStruct[procNr]);
				if (rc == 1) badRead = 1;
			}
		}
		if (badRead == 1) return;
		nJobs = recon_info_record.n_slices * recon_info_record.n_shifts;
		numProcs = (nJobs/2 < numProcs) ? nJobs/2 : numProcs;
		int nrSlicesThread = (int)ceil((double)nJobs / (2.0*(double)numProcs));
		printf("Number of FFT jobs per thread %d, Number of threads: %d.\nStarting processing.\n",nrSlicesThread,numProcs);
		# pragma omp parallel num_threads(numProcs)
		{
			int procNr = omp_get_thread_num();
			int startJobNr, endJobNr;
			startJobNr = procNr*nrSlicesThread*2;
			endJobNr = (startJobNr + nrSlicesThread*2 < nJobs) ? startJobNr + nrSlicesThread*2 : nJobs;
			LOCAL_CONFIG_OPTS information;
			information.shift = recon_info_record.shift_values[0];
			setSinoSize(&information,recon_info_record);
			gridrecParams param;
			param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2;
			param.theta_list = recon_info_record.theta_list;
			param.filter_type = recon_info_record.filter;
			param.theta_list_size = recon_info_record.theta_list_size;
			param.wisdom_string = (char *) malloc(sizeof(char) * (strlen(recon_info_record.wisdom_string)+1));
			param.setPlan = 0;
			strcpy(param.wisdom_string,recon_info_record.wisdom_string);
			size_t offt, offsetRecons;
			setGridRecPSWF(&param);
			initFFTMemoryStructures(&param);
			initGridRec(&param);
			int jobNr, sliceNr, shiftNr, localSliceNr;
			for (jobNr = 0; jobNr < (endJobNr-startJobNr)/2; jobNr ++){
				memsets(&information,recon_info_record);
				sliceNr = (startJobNr + jobNr*2) / recon_info_record.n_shifts;
				shiftNr = (startJobNr + jobNr*2) % recon_info_record.n_shifts;
				localSliceNr = recon_info_record.slices_to_process[sliceNr];
				information.shift = recon_info_record.shift_values[shiftNr];
				memcpy(information.sino_calc_buffer,readStruct[sliceNr].norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				offt = 0;
				offsetRecons = 0;
				reconCentering(&information,recon_info_record,offt,recon_info_record.doLogProj);
				setSinoAndReconBuffers(1, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				information.shift = recon_info_record.shift_values[shiftNr+1];
				memcpy(information.sino_calc_buffer,readStruct[sliceNr].norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				offt = information.sinogram_adjusted_size*2;
				offsetRecons = information.reconstruction_size*4;
				reconCentering(&information,recon_info_record,offt,recon_info_record.doLogProj);
				setSinoAndReconBuffers(2, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				reconstruct(&param);
				information.shift = recon_info_record.shift_values[shiftNr];
				getRecons(&information,recon_info_record,&param,0);
				int rw = writeRecon(localSliceNr,&information,recon_info_record,shiftNr);
				if (rw == 1) continue;
				information.shift = recon_info_record.shift_values[shiftNr+1];
				getRecons(&information,recon_info_record,&param,offsetRecons);
				rw = writeRecon(localSliceNr,&information,recon_info_record,shiftNr+1);
				if (rw == 1) continue;
			}
		}
	}
	double time = omp_get_wtime() - start_time;
	printf("Finished, time elapsed: %lf seconds.\n",time);
	return 0;
}
