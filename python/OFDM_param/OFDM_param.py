import sys
import numpy as np
import json
import argparse
import random
import time

import os
from glob import glob

from getOFDM_param import *
sys.path.append('../')
from utilities import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputJson", help="input JSON filename")
    args = parser.parse_args()
    inputJsonFileName = args.inputJson

    inputJsonFile = open("../inputJson/" + inputJsonFileName + ".json")
    dataPath = json.load(inputJsonFile)["data_path"]
    inputJsonFile.close()
    # dirFilenameList = glob(data_path + '/*.32cf')

    SNRVec = np.arange(0, 21, 2, dtype=int)
    # SNRVec = np.arange(100, 101, 2, dtype=int)
    tauVec = np.array([64, 256, 333, 667, 1333])
    protocolList = np.array(['wlanHT', 'wlanHE', 'NRDLa', 'NRDLb', 'NRDLc'])

    CPLenList = np.array([np.array(['Extended', 'Normal']),\
        np.array(['Extended', 'Medium', 'Normal']),\
        np.array(['Extended', 'Normal']),\
        np.array(['Normal']), np.array(['Normal'])], dtype=object)
    symLenList = np.array([[80, 72], [320, 288, 272], [640, 548], [1096], [2192]], dtype=object)
    
    preambleLen = 1200
    maxInputLen = 20000
    
    fileCount = np.zeros((len(protocolList),), dtype=object)
    corrNSubCCount = np.zeros((len(protocolList),), dtype=object)
    corrSymLenCount = np.zeros((len(protocolList),), dtype=object)
    prevSaveFileName = "../result/start.npy"
    for i in range(len(protocolList)):
        fileCount[i] = np.zeros((len(SNRVec), CPLenList[i].shape[0]))
        corrNSubCCount[i] = np.zeros((len(SNRVec), ))
        corrSymLenCount[i] = np.zeros((len(SNRVec), CPLenList[i].shape[0]))

    countSave = np.concatenate((np.expand_dims(fileCount, axis=0),\
            np.expand_dims(corrNSubCCount, axis=0),\
            np.expand_dims(corrSymLenCount, axis=0)), axis=0)
    with open(prevSaveFileName, 'wb') as saveFile:
        np.save(saveFile, countSave)

    for subFolder in os.walk(dataPath):
        startTime = time.time()
        dirFilenameList = glob(subFolder[0] + '/*.32cf')
        if len(dirFilenameList) == 0:
            continue

        print(subFolder[0].split("/")[-2] + "_" + subFolder[0].split("/")[-1])

        for dirFileName in dirFilenameList:
            fileName = dirFileName.split("/")[-1][:-5]
            fileNameSplit = fileName.split("_")
            fileProtocol = fileNameSplit[0]
            fileCPLen = fileNameSplit[1]
            fileSNR = int(fileNameSplit[4])

            fileProtocolIndex = np.argwhere(fileProtocol == protocolList)[0][0]
            fileSymLen = symLenList[fileProtocolIndex]\
                [np.argwhere(fileCPLen == CPLenList[fileProtocolIndex])[0][0]]
            fileSymLenIndex = np.argwhere(fileCPLen == CPLenList[fileProtocolIndex])[0][0]
            fileSNRIndex = np.argwhere(SNRVec == fileSNR)[0][0]
            # print(fileProtocol, fileCPLen, fileSNR, fileSymLen)

            load_out = np.fromfile(dirFileName, dtype=np.float32)
            data = load_out[np.arange(0, load_out.shape[0], 2)] +\
                                1j * load_out[np.arange(1, load_out.shape[0], 2)]
            inputIQ = data[preambleLen:]
            if inputIQ.shape[0] > maxInputLen:
                randStartIndex = random.randint(0, inputIQ.shape[0] - maxInputLen)
                inputIQ = inputIQ[randStartIndex : randStartIndex + maxInputLen]

            nSubC_Est, symLenEst = getOFDM_param(inputIQ)
            
            fileCount[fileProtocolIndex][fileSNRIndex, fileSymLenIndex] += 1
            if nSubC_Est == tauVec[fileProtocolIndex]:
                corrNSubCCount[fileProtocolIndex][fileSNRIndex] += 1
            if nSubC_Est == tauVec[fileProtocolIndex] and symLenEst == fileSymLen:
                corrSymLenCount[fileProtocolIndex][fileSNRIndex, fileSymLenIndex] += 1
        
        # print(fileCount)
        # print(corrNSubCCount)
        # print(corrSymLenCount)
        print('running time: %s sec' %(time.time() - startTime))

        countSave = np.concatenate((np.expand_dims(fileCount, axis=0),\
            np.expand_dims(corrNSubCCount, axis=0),\
            np.expand_dims(corrSymLenCount, axis=0)), axis=0)
        print(countSave)
        saveFileName = '../result/' + subFolder[0].split("/")[-2] + "_" +\
            subFolder[0].split("/")[-1] + ".npy"
        os.rename(prevSaveFileName, saveFileName)
        with open(saveFileName, 'wb') as saveFile:
            np.save(saveFile, countSave)
        prevSaveFileName = saveFileName

if __name__ == "__main__":
    main()