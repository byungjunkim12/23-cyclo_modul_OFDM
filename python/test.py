import sys
import numpy as np
import json
import argparse
import random
import time
from scipy import signal

import os
from glob import glob

from OFDMparam import *
from CFOcorr import *
from getModFeat import *
from utilities import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputJson", help="input JSON filename")
    parser.add_argument("-c", "--classes", help="class JSON filename")
    parser.add_argument("-t", "--train", help="train")
    args = parser.parse_args()
    inputJsonFileName = args.inputJson
    classJsonFileName = args.classes
    trainFlag = args.train

    inputJsonFile = open("./inputJson/" + inputJsonFileName + ".json")
    classJsonFile = open("./inputJson/class/" + classJsonFileName + ".json")
    classJson = json.load(classJsonFile)

    dataPath = json.load(inputJsonFile)["data_path"]

    SNRVec = np.asarray(classJson["SNRVec"])
    tauVec = np.asarray(classJson["tauVec"])
    protocolList = np.asarray(classJson["protocolList"])
    CPOptList = np.asarray(classJson["CPOptList"], dtype=object)
    for i, CPLenElem in enumerate(CPOptList):
        CPOptList[i] = np.asarray(CPLenElem, dtype=object)
    CPLenList = np.asarray(classJson["CPLenList"], dtype=object)
    
    inputJsonFile.close()
    classJsonFile.close()
    
    longestPreambleLen = 1200
    maxInputLen = 300000
    wlanHTInputSym = 40
    wlanHEInputSym = 10

    nFFT = 4096
    samplingRate = 20e6
    NRDLTXRate = 30.72e6
    inputNHalfSubfr = 6
    lenHalfSubfr = int(0.5e-3 * NRDLTXRate)

    fileCount = np.zeros((len(protocolList),), dtype=object)
    corrNSubCCount = np.zeros((len(protocolList),), dtype=object)
    corrCPLenCount = np.zeros((len(protocolList),), dtype=object)
    resultList = []
    for i, protocol in enumerate(protocolList):
        fileCount[i] = np.zeros((len(SNRVec), CPOptList[i].shape[0]))
        corrNSubCCount[i] = np.zeros((len(SNRVec), ))
        corrCPLenCount[i] = np.zeros((len(SNRVec), CPOptList[i].shape[0]))
        if "NRDL" in protocol:
            tauVec[i] = (tauVec[i] * samplingRate / NRDLTXRate).astype(int)
            for j, CPLenElem in enumerate(CPOptList[i]):
                CPLenList[i][j] = round(CPLenList[i][j] * samplingRate / NRDLTXRate)

    saveFileNamePrefix = getDateYYMMDD_HHMMSS()
    os.mkdir("./result/" + saveFileNamePrefix)
    saveLogFileName = "./result/" + saveFileNamePrefix + "/log.txt"
    saveResultFileName = "./result/" + saveFileNamePrefix + "/result.npy"
    print("Save folder name: ", saveFileNamePrefix)
    print(dataPath)

    for subFolder in os.walk(dataPath):
        startTime = time.time()
        dirFilenameList = glob(subFolder[0] + '/*.32cf')
        resultMat = np.zeros((len(dirFilenameList), 7), dtype=object)
        if len(dirFilenameList) == 0:
            continue
        
        print('Folder: ',subFolder[0].split("/")[-1],', # of Files:',len(dirFilenameList))
        
        for fileIndex, dirFileName in enumerate(dirFilenameList):
            fileName = dirFileName.split("/")[-1][:-5]
            resultMat[fileIndex, 0] = fileName

            load_out = np.fromfile(dirFileName, dtype=np.float32)
            data = load_out[np.arange(0, load_out.shape[0], 2)] +\
                                1j * load_out[np.arange(1, load_out.shape[0], 2)]
            inputIQ = data[longestPreambleLen:]
            if inputIQ.shape[0] > maxInputLen:
                randStartIndex = random.randint(0, inputIQ.shape[0] - maxInputLen)
                inputIQ = inputIQ[randStartIndex : randStartIndex + maxInputLen]
            nSubC_Est, CPLenEst = getOFDM_param(inputIQ, protocolList, tauVec, nFFT, CPLenList)
            
            resultMat[fileIndex, 1] = nSubC_Est
            resultMat[fileIndex, 2] = CPLenEst

            # if nSubC_Est == 64 or nSubC_Est == 256:
            #     if nSubC_Est == 64:
            #         inputLen = (wlanHTInputSym+2) * (nSubC_Est+CPLenEst)
            #     elif nSubC_Est == 256:
            #         inputLen = (wlanHEInputSym+2) * (nSubC_Est+CPLenEst)      
            #     inputStartIndex = int(random.randint(longestPreambleLen, inputIQ.shape[0]-inputLen))
            #     inputIQ = data[inputStartIndex : inputStartIndex+inputLen]
            #     firstIndexEst = findFirstIndexWifi(inputIQ, nSubC_Est, CPLenEst)  
            #     resultMat[fileIndex, 4] = inputStartIndex
            
            # else:
            #     inputIQRes = signal.resample_poly(data, NRDLTXRate, samplingRate)
            #     if nSubC_Est == 333:
            #         nSubC_Est = 512
            #         if CPLenEst == 23:
            #             addInputLen = 2208 + round((nSubC_Est+CPLenEst)/2)
            #             CPLenEst = 36
            #         else:
            #             addInputLen = 0
            #             CPLenEst = 128   
            #     elif nSubC_Est == 666:
            #         nSubC_Est = 1024
            #         CPLenEst = 72
            #         addInputLen = 4400 + int((nSubC_Est+CPLenEst)/2)
            #     elif nSubC_Est == 1333:
            #         nSubC_Est = 2048
            #         CPLenEst = 144
            #         addInputLen = 8784 + int((nSubC_Est+CPLenEst)/2)
                
            #     inputLen = inputNHalfSubfr * lenHalfSubfr + addInputLen
            #     inputStartIndex = int(random.randint(0, inputIQRes.shape[0]-inputLen))
            #     inputIQ = inputIQRes[inputStartIndex : inputStartIndex+inputLen]
            #     [firstIndexEst, firstSymIndexEst] = findFirstIndexNR(inputIQ, nSubC_Est, CPLenEst)
            #     # [startIndexTrue, symIndexTrue] = findTrueFirstIndexNR(inputStartIndex, nSubC_Est, CPLenEst)
            #     # print(inputStartIndex, firstIndexEst, startIndexTrue)

            #     resultMat[fileIndex, 4] = inputStartIndex
            #     resultMat[fileIndex, 5] = firstSymIndexEst

            # resultMat[fileIndex, 3] = firstIndexEst

            # resultMat[fileIndex, 5] = estCFO(inputIQ, nSubC_Est, CPLenEst, resultMat[fileIndex, 3], samplingRate)
            # inputIQ_CFOcorr = corrCFO(inputIQ, nSubC_Est, CPLenEst, firstIndexEst, samplingRate, 2)

        print('running time: %0.2f sec' %(time.time() - startTime))

        resultList.append(resultMat) 
        with open(saveLogFileName, 'a') as saveLogFile:
            saveLogFile.write(subFolder[0].split("/")[-2] + "/" + subFolder[0].split("/")[-1])
            saveLogFile.write('\n')
        with open(saveResultFileName, 'wb') as saveResultFile:
            np.save(saveResultFile, resultList)


if __name__ == "__main__":
    main()