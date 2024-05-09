#!/usr/bin/env python

import parsl
import subprocess
import sys, os
import time
import argparse
import signal
import shutil
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
v7Dir = os.path.expanduser('~/opt/MIDAS/FF_HEDM/v7/')
sys.path.insert(0,v7Dir)
from parsl.app.app import python_app
pytpath = sys.executable

def generateZip(resFol,pfn,layerNr,dfn='',dloc='',nchunks=-1,preproc=-1,outf='ZipOut.txt',errf='ZipErr.txt'):
    cmd = pytpath+' '+os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZip.py')+' -resultFolder '+ resFol +' -paramFN ' + pfn + ' -LayerNr ' + str(layerNr)
    if dfn!='':
        cmd+= ' -dataFN ' + dfn
    if dloc!='':
        cmd+= ' -dataLoc ' + dloc
    if nchunks!=-1:
        cmd+= ' -numFrameChunks '+str(nchunks)
    if preproc!=-1:
        cmd+= ' -preProcThresh '+str(preproc)
    outf = resFol+'/output/'+outf
    errf = resFol+'/output/'+errf
    subprocess.call(cmd,shell=True,stdout=open(outf,'w'),stderr=open(errf,'w'))
    lines = open(outf,'r').readlines()
    if lines[-1].startswith('OutputZipName'):
        return lines[-1].split()[1]

@python_app
def peaks(resultDir,zipFN,numProcs,blockNr=0,numBlocks=1):
    import subprocess
    import os
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'
    f = open(f'{resultDir}/output/peaksearch_out{blockNr}.csv','w')
    f_err = open(f'{resultDir}/output/peaksearch_err{blockNr}.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMPZarr")+f' {zipFN} {blockNr} {numBlocks} {numProcs}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()

@python_app
def index(resultDir,numProcs,blockNr=0,numBlocks=1):
    import subprocess
    import os
    os.chdir(resultDir)
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'
    with open("SpotsToIndex.csv", "r") as f:
        num_lines = len(f.readlines())
    f = open(f'{resultDir}/output/indexing_out{blockNr}.csv','w')
    f_err = open(f'{resultDir}/output/indexing_err{blockNr}.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerOMP")+f' paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()

@python_app
def refine(resultDir,numProcs,blockNr=0,numBlocks=1):
    import subprocess
    import os
    os.chdir(resultDir)
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'
    with open("SpotsToIndex.csv", "r") as f:
        num_lines = len(f.readlines())
    f = open(f'{resultDir}/output/refining_out{blockNr}.csv','w')
    f_err = open(f'{resultDir}/output/refining_err{blockNr}.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitPosOrStrainsOMP")+f' paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()

default_handler = None

def handler(num, frame):    
    subprocess.call("rm -rf /dev/shm/*.bin",shell=True)
    print("Ctrl-C was pressed, cleaning up.")
    return default_handler(num, frame) 

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

default_handler = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, handler)
parser = MyParser(description='''Far-field HEDM analysis using MIDAS. V7.0.0, contact hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=False, default='', help='Folder where you want to save results. If nothing is provided, it will default to the current folder.')
parser.add_argument('-paramFN', type=str, required=False, default='', help='Parameter file name. Provide either paramFN and/or dataFN.')
parser.add_argument('-dataFN', type=str, required=False, default='', help='Data file name. This is if you have either h5 or zip files.  Provide either paramFN and/or dataFN (in case zip exists).')
parser.add_argument('-nCPUs', type=int, required=False, default=10, help='Number of CPU cores to use if running locally.')
parser.add_argument('-machineName', type=str, required=False, default='local', help='Machine name for execution, local, orthrosnew, orthrosall, umich, marquette, purdue.')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='If low on RAM, it can process parts of the dataset at the time. -1 will disable.')
parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
parser.add_argument('-nNodes', type=int, required=False, default=-1, help='Number of nodes for execution, omit if want to automatically select.')
parser.add_argument('-startLayerNr', type=int, required=False, default=1, help='Start LayerNr to process')
parser.add_argument('-endLayerNr', type=int, required=False, default=1, help='End LayerNr to process')
parser.add_argument('-convertFiles', type=int, required=False, default=1, help='If want to convert to zarr, if zarr files exist already, put to 0.')
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    print("MUST PROVIDE EITHER paramFN or dataFN")
    sys.exit(1)
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
psFN = args.paramFN
dataFN = args.dataFN
numProcs = args.nCPUs
machineName = args.machineName
nNodes = args.nNodes
nchunks = args.numFrameChunks
preproc = args.preProcThresh
startLayerNr = args.startLayerNr
endLayerNr = args.endLayerNr
ConvertFiles = args.convertFiles
if nNodes == -1:
    nNodes = 1

env = dict(os.environ)
midas_path = os.path.expanduser("~/.MIDAS")
env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'
if len(resultDir) == 0 or resultDir=='.':
    resultDir = os.getcwd()
logDir = resultDir + '/output'
os.makedirs(resultDir,exist_ok=True)
os.makedirs(logDir,exist_ok=True)
os.environ['MIDAS_SCRIPT_DIR'] = logDir
if machineName == 'local':
    nNodes = 1
    from localConfig import *
    parsl.load(config=localConfig)
elif machineName == 'orthrosnew':
    numProcs = 32
    nNodes = 11
    from orthrosAllConfig import *
    parsl.load(config=orthrosNewConfig)
elif machineName == 'orthrosall':
    numProcs = 64
    nNodes = 5
    from orthrosAllConfig import *
    parsl.load(config=orthrosAllConfig)
elif machineName == 'umich':
    numProcs = 36
    os.environ['nNodes'] = str(nNodes)
    from uMichConfig import *
    parsl.load(config=uMichConfig)
elif machineName == 'marquette':
    numProcs = 36
    os.environ['nNodes'] = str(nNodes)
    from marquetteConfig import *
    parsl.load(config=marquetteConfig)
elif machineName == 'purdue':
    numProcs = 128
    os.environ['nNodes'] = str(nNodes)
    from purdueConfig import *
    parsl.load(config=purdueConfig)

# Run for each layer.
origDir = os.getcwd()
topResDir = resultDir
for layerNr in range(startLayerNr,endLayerNr+1):
    resultDir = f'{topResDir}/LayerNr_{layerNr}'
    print(f"Doing Layer Nr: {layerNr}, results will be saved in {resultDir}")
    logDir = resultDir + '/output'
    os.makedirs(resultDir,exist_ok=True)
    os.makedirs(logDir,exist_ok=True)
    t0 = time.time()
    if ConvertFiles == 1:
        if len(dataFN)>0:
            print("Generating combined MIDAS file from HDF and ps files.")
        else:
            print("Generating combined MIDAS file from GE and ps files.")
        outFStem = generateZip(resultDir,psFN,layerNr,dfn=dataFN,nchunks=nchunks,preproc=preproc)
    else:
        if len(dataFN) > 0:
            outFStem = f'{resultDir}/{dataFN}'
            if not os.path.exists(outFStem):
                shutil.copy2(dataFN,resultDir)
        else:
            psContents = open(psFN).readlines()
            for line in psContents:
                if line.startswith('FileStem '):
                    fStem = line.split()[1]
                if line.startswith('StartFileNrFirstLayer '):
                    startFN = int(line.split()[1])
                if line.startswith('NrFilesPerSweep '):
                    NrFilerPerLayer = int(line.split()[1])
            thisFileNr = startFN + (layerNr-1)*NrFilerPerLayer
            outFStem = f'{resultDir}/{fStem}_{str(thisFileNr).zfill(6)}.MIDAS.zip'
            if not os.path.exists(outFStem):
                shutil.copy2(dataFN,resultDir)
        cmdUpd = f'{pytpath} ' + os.path.expanduser('~/opt/MIDAS/utils/updateZarrDset.py')
        cmdUpd += f' -fn {os.path.basename(outFStem)} -folder {resultDir} -keyToUpdate ResultFolder -updatedValue {resultDir}/'
        subprocess.call(cmdUpd,shell=True)
        print(outFStem)
    print(f"Generating HKLs. Time till now: {time.time()-t0} seconds.")
    f_hkls = open(f'{logDir}/hkls_out.csv','w')
    f_hkls_err = open(f'{logDir}/hkls_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLListZarr")+' '+outFStem,shell=True,env=env,stdout=f_hkls,stderr=f_hkls_err)
    f_hkls.close()
    f_hkls_err.close()
    os.makedirs(f'{resultDir}/Temp')
    print(f"Doing PeakSearch. Time till now: {time.time()-t0} seconds.")
    res = []
    for nodeNr in range(nNodes):
        res.append(peaks(resultDir,outFStem,numProcs,blockNr=nodeNr,numBlocks=nNodes))
    outputs = [i.result() for i in res]
    print(f"Merging peaks. Time till now: {time.time()-t0}")
    f = open(f'{logDir}/merge_overlaps_out.csv','w')
    f_err = open(f'{logDir}/merge_overlaps_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAllZarr")+' '+outFStem,shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()
    print(f"Calculating Radii. Time till now: {time.time()-t0}")
    f = open(f'{logDir}/calc_radius_out.csv','w')
    f_err = open(f'{logDir}/calc_radius_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAllZarr")+' '+outFStem,shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()
    print(f"Transforming data. Time till now: {time.time()-t0}")
    f = open(f'{logDir}/fit_setup_out.csv','w')
    f_err = open(f'{logDir}/fit_setup_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetupZarr")+' '+outFStem,shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()
    os.chdir(resultDir)
    print(f"Binning data. Time till now: {time.time()-t0}")
    f2 = open(f'{logDir}/binning_out.csv','w')
    f_err2 = open(f'{logDir}/binning_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinData")+' paramstest.txt',shell=True,env=env,stdout=f2,stderr=f_err2)
    f2.close()
    f_err2.close()
    print(f"Indexing. Time till now: {time.time()-t0}")
    resIndex = []
    for nodeNr in range(nNodes):
        resIndex.append(index(resultDir,numProcs,blockNr=nodeNr,numBlocks=nNodes))
    outputIndex = [i.result() for i in resIndex]
    print(f"Refining. Time till now: {time.time()-t0}")
    resRefine = []
    for nodeNr in range(nNodes):
        resRefine.append(refine(resultDir,numProcs,blockNr=nodeNr,numBlocks=nNodes))
    outputRefine = [i.result() for i in resRefine]
    subprocess.call("rm -rf /dev/shm/*.bin",shell=True)
    print(f"Making grains list. Time till now: {time.time()-t0}")
    f = open(f'{logDir}/process_grains_out.csv','w')
    f_err = open(f'{logDir}/process_grains_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/ProcessGrainsZarr")+' '+outFStem,shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()
    # print(f"Making plots, condensing output. Time till now: {time.time()-t0}")
    # subprocess.call(f'{pytpath} '+os.path.expanduser('~/opt/MIDAS/utils/plotFFSpots3d.py')+' -resultFolder '+resultDir,cwd=resultDir, shell=True)
    # subprocess.call(f'{pytpath} '+os.path.expanduser('~/opt/MIDAS/utils/plotFFSpots3dGrains.py')+' -resultFolder '+resultDir,cwd=resultDir,shell=True)
    # subprocess.call(f'{pytpath} '+os.path.expanduser('~/opt/MIDAS/utils/plotGrains3d.py')+' -resultFolder '+resultDir,cwd=resultDir,shell=True)
    print(f"Done Layer {layerNr}. Total time elapsed: {time.time()-t0}")
    os.chdir(origDir)
