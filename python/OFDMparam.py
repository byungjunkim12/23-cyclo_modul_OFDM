import numpy as np

def getOFDM_param(inputIQ, protocolList, tauVec, nFFT, CPLenList):
    nFFT = 4096
    samplingRate = 20e6
    NRDLTXRate = 30.72e6
    inputLen = 2

    freqBinList = np.zeros((len(protocolList),), dtype=object)
    for i, protocol in enumerate(protocolList):
        if "wlan" in protocol:
            freqBinList[i] = np.round(nFFT / (tauVec[i] + CPLenList[i])).astype(int)
        elif "NRDLa" in protocol:
            freqBinList[i] = np.round(nFFT / (tauVec[i] + CPLenList[i])).astype(int)
            # freqBinList[i] = np.round(nFFT / (tauVec[i] + np.asarray(CPLenList[i]) * samplingRate / NRDLTXRate)).astype(int)
        else:
            freqBinList[i] = [0]
    # print(freqBinList)
    
    CAF_DC = getCAF_DC(inputIQ, tauVec, inputLen)
    nSubC_Est = tauVec[np.argmax(CAF_DC)]
    CAF = getCAF(inputIQ, nFFT, nSubC_Est, inputLen)
    CPLenEst = CPLenList[np.argmax(CAF_DC)][np.argmax(CAF[freqBinList[np.argmax(CAF_DC)]])]

    return nSubC_Est, CPLenEst

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
    SpecStep = 10

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
