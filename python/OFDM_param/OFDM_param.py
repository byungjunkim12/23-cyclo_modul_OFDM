import sys
import numpy as np
import json
import argparse
from glob import glob

sys.path.append('../')

from utilities import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputJson", help="input JSON filename")
    args = parser.parse_args()

    inputJsonFileName = args.inputJson

    inputJsonFile = open("../inputJson/" + inputJsonFileName + ".json")
    data_path = json.load(inputJsonFile)["data_path"]
    dirFilenameList = glob(data_path + '/*.32cf')
    
    SNRVec = np.arange(0, 21, 2, dtype=int)
    # SNRVec = np.arange(100, 101, 2, dtype=int)
    FFTsize = 4096
    tauVec = np.array([64, 256, 333, 667, 1333])
    CPLenList = np.array([np.array(['Extended', 'Normal']),\
        np.array(['Extended', 'Medium', 'Normal']),\
        np.array(['Extended', 'Normal'])])
    freqBinList = [[51, 57], [13, 14, 15], [10, 12]]
    symLenList = np.array([[80, 72], [320, 288, 272], [640, 548]])
    protocolList = np.array(['wlanHT', 'wlanHE', 'NRDLa', 'NRDLb', 'NRDLc'])
    inputLen = 8
    preambleLen = 1200
    print(len(protocolList))
    
    fileCount = np.zeros((len(SNRVec), len(protocolList)))
    corrCount = np.zeros((len(SNRVec), len(protocolList)))
    for dirFileName in dirFilenameList:
        # dirFileNameSplit = dirFileName.split("/")
        fileName = dirFileName.split("/")[-1][:-5]
        fileNameSplit = fileName.split("_")
        fileProtocol = fileNameSplit[0]
        fileCPLen = fileNameSplit[1]
        fileSNR = int(fileNameSplit[4])

        fileProtocolIndex = np.argwhere(fileProtocol == protocolList)[0][0]
        fileSymLen = symLenList[fileProtocolIndex]\
            [np.argwhere(fileCPLen == CPLenList[fileProtocolIndex])[0][0]]
        # print(fileProtocol, fileCPLen, fileSNR, fileSymLen)

        load_out = np.fromfile(dirFileName, dtype=np.float32)
        data = load_out[np.arange(0, load_out.shape[0], 2)] +\
                            1j * load_out[np.arange(1, load_out.shape[0], 2)]
        inputIQ = data[preambleLen:]

        CAF_DC = getCAF_DC(inputIQ, tauVec, inputLen)
        nSubC_Est = tauVec[np.argmax(CAF_DC)]
        CAF = getCAF(inputIQ, FFTsize, nSubC_Est, inputLen)
        symLenEst = symLenList[np.argmax(CAF_DC)][np.argmax(CAF[freqBinList[np.argmax(CAF_DC)]])]
        
        fileCount[np.argwhere(SNRVec == fileSNR)[0][0], fileProtocolIndex] += 1        
        if fileProtocolIndex == np.argmax(CAF_DC) and\
            np.argwhere(fileCPLen == CPLenList[fileProtocolIndex])[0][0] ==\
            np.argmax(CAF[freqBinList[np.argmax(CAF_DC)]]):
            corrCount[np.argwhere(SNRVec == fileSNR)[0][0], fileProtocolIndex] += 1

        # print(np.argwhere(SNRVec == fileSNR)[0][0], np.argwhere(protocolList == fileProtocol)[0][0])
        # print(fileName, nSubC_Est, symLenEst)
    print(fileCount)
    print(corrCount)
if __name__ == "__main__":
    main()