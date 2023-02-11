import numpy as np

def getMCS_HEMU(inputJson):
    nUser = len(inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'])
    MCS_list = np.zeros((nUser,), dtype=int)
    for iUser in range(nUser):
        MCS_list[iUser] = inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'][iUser]['MCS']

    return MCS_list

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

    if CPLen == 1:
        inputCorr = inputMat[:-tau] * np.conj(inputMat[tau:])
    else:
        inputCorr = np.sum(inputMat[:, :-tau] * np.conj(inputMat[:, tau:]), axis=0)

    inputSpec = np.zeros((FFTsize, len(inputCorr) - FFTsize + 1))
    for i in range(inputSpec.shape[1]):
        inputSpec[:, i] = np.fft.fft(inputCorr[i : i+FFTsize])
    CAF = np.mean(np.abs(inputSpec), axis=1)
    return CAF


