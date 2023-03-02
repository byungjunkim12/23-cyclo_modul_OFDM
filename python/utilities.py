import numpy as np
from scipy.signal import find_peaks

def getMCS_HEMU(inputJson):
    nUser = len(inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'])
    MCS_list = np.zeros((nUser,), dtype=int)
    for iUser in range(nUser):
        MCS_list[iUser] = inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'][iUser]['MCS']

    return MCS_list

def findFirstIndex(inputIQ, nSubC, CPLen):
    indexMat = np.arange(CPLen)[:, np.newaxis] + np.arange(inputIQ.shape[0]-CPLen+1)
    inputMat = inputIQ[indexMat]
    
    inputCorr = np.sum(inputMat[:, :-nSubC] * np.conj(inputMat[:, nSubC:]), axis=0)
    peaks, pklocs = find_peaks(np.abs(inputCorr), distance=np.floor(0.9*(nSubC+CPLen)))
    print(np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen))
    firstIndex = np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen)

    return firstIndex