import numpy as np
from scipy import signal

def getOFDM_param(inputIQ):
    FFTsize = 4096
    tauVec = np.array([64, 256, 333, 667, 1333])
    freqBinList = [[51, 57], [13, 14, 15], [10, 12], [0], [0]]
    symLenList = np.array([[80, 72], [320, 288, 272], [640, 548], [1096], [2192]], dtype=object)
    inputLen = 8
    
    CAF_DC = getCAF_DC(inputIQ, tauVec, inputLen)
    nSubC_Est = tauVec[np.argmax(CAF_DC)]
    CAF = getCAF(inputIQ, FFTsize, nSubC_Est, inputLen)
    symLenEst = symLenList[np.argmax(CAF_DC)][np.argmax(CAF[freqBinList[np.argmax(CAF_DC)]])]

    return nSubC_Est, symLenEst

def getCAF_DC(inputIQ, tauVec, CPLen):
    indexMat = np.arange(1, CPLen + 1).reshape(-1, 1) + np.arange(0, len(inputIQ) - CPLen)
    inputMat = inputIQ[indexMat]
    CAF_DC = np.zeros(len(tauVec))

    for tauIndex in range(len(tauVec)):
        tau = tauVec[tauIndex]
        if CPLen == 1:
            inputCorr = inputMat[:-tau] * np.conj(inputMat[tau:])
        else:
            inputCorr = np.sum(inputMat[:, :-tau] * np.conj(inputMat[:, tau:]), axis=1)
        CAF_DC[tauIndex] = np.mean(np.abs(inputCorr))

    return CAF_DC

def getCAF(inputIQ, FFTsize, tau, CPLen):
    indexMat = np.arange(1, CPLen + 1).reshape(-1, 1) + np.arange(0, len(inputIQ) - CPLen)
    inputMat = inputIQ[indexMat]
    SpecStep = 7

    if CPLen == 1:
        inputCorr = inputMat[:-tau] * np.conj(inputMat[tau:])
    else:
        inputCorr = np.sum(inputMat[:, :-tau] * np.conj(inputMat[:, tau:]), axis=0)

    inputSpecRange = np.arange(0, len(inputCorr) - FFTsize, SpecStep)
    inputSpec = np.zeros((FFTsize, inputSpecRange.shape[0]), dtype=complex)
    for i in range(inputSpecRange.shape[0]):
        corrIndex = inputSpecRange[i]
        inputSpec[:, i] = np.fft.fft(inputCorr[corrIndex : corrIndex+FFTsize])
    CAF = np.mean(np.abs(inputSpec), axis=1)
    return CAF
