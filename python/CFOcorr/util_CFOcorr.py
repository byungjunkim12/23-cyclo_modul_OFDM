import math
import numpy as np
from scipy.signal import find_peaks

def findFirstIndex(inputIQ, nSubC, CPLen):
    indexMat = np.arange(CPLen)[:, np.newaxis] + np.arange(inputIQ.shape[0]-CPLen+1)
    inputMat = inputIQ[indexMat]
    
    inputCorr = np.sum(inputMat[:, :-nSubC] * np.conj(inputMat[:, nSubC:]), axis=0)
    peaks, pklocs = find_peaks(np.abs(inputCorr), distance=np.floor(0.9*(nSubC+CPLen)))
    print(np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen))
    firstIndex = np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen)

    return firstIndex

def estCFO(inputIQ, nSubC, lenCP, firstIndexSym, samplingRate):
    lenOFDMSym = nSubC + lenCP
    nOFDMSym = int(np.floor((inputIQ[firstIndexSym + int(lenCP/4) : ].shape)[0] / lenOFDMSym))
    phDiffSym = np.zeros((nOFDMSym-1, ))

    for iSym in range(nOFDMSym-1):  
        tempPhDiffSym = np.angle(inputIQ[firstIndexSym + int(lenCP/4) + iSym*lenOFDMSym + nSubC :\
                firstIndexSym + int(lenCP/4) + iSym*lenOFDMSym + nSubC + int(lenCP/2)]) -\
                np.angle(inputIQ[firstIndexSym + int(lenCP/4) + iSym*lenOFDMSym :\
                firstIndexSym + int(lenCP/4) + iSym*lenOFDMSym + int(lenCP/2)])

        tempPhDiffSym[tempPhDiffSym > math.pi] = tempPhDiffSym[tempPhDiffSym > math.pi] - 2*math.pi
        tempPhDiffSym[tempPhDiffSym < -math.pi] = tempPhDiffSym[tempPhDiffSym < -math.pi] + 2*math.pi
        phDiffSym[iSym] = np.mean(tempPhDiffSym)

    CFOest = np.mean(phDiffSym) / (2*math.pi * nSubC) * samplingRate

    return CFOest

def corrCFO(inputIQ, nSubC, lenCP, firstIndexSym, samplingRate, nIter):
    for _ in range(nIter):
        CFOest = estCFO(inputIQ, nSubC, lenCP, firstIndexSym, samplingRate)
        inputIQ = np.multiply(inputIQ, np.exp(-1j*2*math.pi*CFOest/samplingRate*np.arange(inputIQ.shape[0])))
    
    return inputIQ