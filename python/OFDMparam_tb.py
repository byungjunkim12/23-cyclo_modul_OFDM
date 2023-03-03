import sys
import numpy as np
import json
import argparse
import random
import time

import os
from glob import glob

from OFDMparam import *
sys.path.append('../')
from utilities import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classes", help="class JSON filename")
    parser.add_argument("-i", "--inputJson", help="input JSON filename")
    args = parser.parse_args()
    inputJsonFileName = args.inputJson
    classesJsonFileName = args.classes

    inputJsonFile = open("../inputJson/" + inputJsonFileName + ".json")
    classesJsonFile = open("../inputJson/classes/" + classesJsonFileName + ".json")
    classesJson = json.load(classesJsonFile)

    dataPath = json.load(inputJsonFile)["data_path"]

    SNRVec = np.asarray(classesJson["SNRVec"])
    tauVec = np.asarray(classesJson["tauVec"])
    protocolList = np.asarray(classesJson["protocolList"])
    CPOptList = np.asarray(classesJson["CPOptList"])
    for i, CPLenElem in enumerate(CPOptList):
        CPOptList[i] = np.asarray(CPLenElem, dtype=object)
    CPLenList = np.asarray(classesJson["CPLenList"], dtype=object)
    
    inputJsonFile.close()
    classesJsonFile.close()
    
    preambleLen = 1200
    maxInputLen = 20000
    nFFT = 4096
    samplingRate = 20e6
    NRDLTXRate = 30.72e6

    fileCount = np.zeros((len(protocolList),), dtype=object)
    corrNSubCCount = np.zeros((len(protocolList),), dtype=object)
    corrCPLenCount = np.zeros((len(protocolList),), dtype=object)

    for i, protocol in enumerate(protocolList):
        fileCount[i] = np.zeros((len(SNRVec), CPOptList[i].shape[0]))
        corrNSubCCount[i] = np.zeros((len(SNRVec), ))
        corrCPLenCount[i] = np.zeros((len(SNRVec), CPOptList[i].shape[0]))        
        if "NRDL" in protocol:
            tauVec[i] = (tauVec[i] * samplingRate / NRDLTXRate).astype(int)

    countSave = np.concatenate((np.expand_dims(fileCount, axis=0),\
            np.expand_dims(corrNSubCCount, axis=0),\
            np.expand_dims(corrCPLenCount, axis=0)), axis=0)

    saveFileNamePrefix = getDateYYMMDD_HHMMSS()
    saveFileName = "../result/" + saveFileNamePrefix + ".npy"
    saveLogFileName = "../result/log/" + saveFileNamePrefix + ".txt"
    print("Filename: ", saveFileNamePrefix)

    for subFolder in os.walk(dataPath):
        startTime = time.time()
        dirFilenameList = glob(subFolder[0] + '/*.32cf')
        if len(dirFilenameList) == 0:
            continue

        print(subFolder[0].split("/")[-2] + "/" + subFolder[0].split("/")[-1])

        for dirFileName in dirFilenameList:
            fileName = dirFileName.split("/")[-1][:-5]
            fileNameSplit = fileName.split("_")
            fileProtocol = fileNameSplit[0]
            fileCPOpt = fileNameSplit[1]
            fileSNR = int(fileNameSplit[4])

            fileProtocolIndex = np.argwhere(fileProtocol == protocolList)[0][0]
            fileCPLen = CPLenList[fileProtocolIndex]\
                [np.argwhere(fileCPOpt == CPOptList[fileProtocolIndex])[0][0]]
            fileCPOptIndex = np.argwhere(fileCPOpt == CPOptList[fileProtocolIndex])[0][0]
            fileSNRIndex = np.argwhere(SNRVec == fileSNR)[0][0]
            # print(fileProtocol, fileCPOpt, fileSNR, fileCPLen)

            load_out = np.fromfile(dirFileName, dtype=np.float32)
            data = load_out[np.arange(0, load_out.shape[0], 2)] +\
                                1j * load_out[np.arange(1, load_out.shape[0], 2)]
            inputIQ = data[preambleLen:]
            if inputIQ.shape[0] > maxInputLen:
                randStartIndex = random.randint(0, inputIQ.shape[0] - maxInputLen)
                inputIQ = inputIQ[randStartIndex : randStartIndex + maxInputLen]

            nSubC_Est, CPLenEst = getOFDM_param(inputIQ, protocolList, tauVec, nFFT, CPLenList)
            
            fileCount[fileProtocolIndex][fileSNRIndex, fileCPOptIndex] += 1
            if nSubC_Est == tauVec[fileProtocolIndex]:
                corrNSubCCount[fileProtocolIndex][fileSNRIndex] += 1
            if nSubC_Est == tauVec[fileProtocolIndex] and CPLenEst == fileCPLen:
                corrCPLenCount[fileProtocolIndex][fileSNRIndex, fileCPOptIndex] += 1
        
        print('running time: %s sec' %(time.time() - startTime))

        countSave = np.concatenate((np.expand_dims(fileCount, axis=0),\
            np.expand_dims(corrNSubCCount, axis=0),\
            np.expand_dims(corrCPLenCount, axis=0)), axis=0)
        print(countSave)
        with open(saveFileName, 'wb') as saveFile:
            np.save(saveFile, countSave)
        with open(saveLogFileName, 'a') as saveLogFile:
            saveLogFile.write(subFolder[0].split("/")[-2] + "/" + subFolder[0].split("/")[-1])
            saveLogFile.write('\n')

if __name__ == "__main__":
    main()