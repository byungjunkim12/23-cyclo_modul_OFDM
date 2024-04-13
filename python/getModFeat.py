from cmath import pi
import numpy as np
import math
from utilities import *

pi = math.pi


def getModFeatWifi(inputIQ, nSubC, CPLen, firstIndexSym, nSym, removeNull, angleMod):
    featAbs = np.zeros((nSubC, nSym))
    featPh = np.zeros((nSubC, nSym))
    if nSubC == 64:
        nNullSubC = 8
    elif nSubC == 256:
        nNullSubC = 32

    for iSym in range(nSym):
        symbolsFreq = np.fft.fft(inputIQ[(firstIndexSym+int(CPLen/2) + iSym*(nSubC+CPLen)) :\
            (firstIndexSym+nSubC+int(CPLen/2) + iSym*(nSubC+CPLen))])
        nextSymbolsFreq = np.fft.fft(inputIQ[(firstIndexSym+int(CPLen/2) + (iSym+1)*(nSubC+CPLen)) :\
            (firstIndexSym+nSubC+int(CPLen/2) + (iSym+1)*(nSubC+CPLen))])

        featAbs[:, iSym] = np.abs(symbolsFreq)
        tempPhDiffSym = np.angle(nextSymbolsFreq) - np.angle(symbolsFreq)
        tempPhDiffSym[tempPhDiffSym > pi] = tempPhDiffSym[tempPhDiffSym > pi] - 2*pi
        tempPhDiffSym[tempPhDiffSym < -pi] = tempPhDiffSym[tempPhDiffSym < -pi] + 2*pi
        featPh[:, iSym] = tempPhDiffSym

        if angleMod:
            featPh[:, iSym] = np.mod(featPh[:, iSym], (pi/2))
            # tempFeatPh = featPh[:, iSym]
            # tempFeatPh[tempFeatPh > pi/4] = pi/2 - tempFeatPh[tempFeatPh > pi/4]
            # featPh[:, iSym] = tempFeatPh

    feat = np.multiply(featAbs, np.exp(1j*featPh))
    if removeNull:
        sortSubC = np.argsort(np.mean(featAbs, axis=1))
        subCwoNull = sortSubC[nNullSubC:]
        feat = feat[subCwoNull, :]
        
    return feat
    
def getModFeatNR(inputIQ, indexNRDL, nSubC, CPLen, nSym, angleMod, firstIndex, firstSymIndex):
    nFreqRB = 12
    chIndexSym = []

    if nSubC == 512 and CPLen == 128:
        nSymHalfSubfr = 24
        maxnRB = 24
        indexTranMat = [368, 512, 0, 144]
    elif nSubC == 512 and CPLen == 36:
        nSymHalfSubfr = 28
        maxnRB = 24
        indexTranMat = [368, 512, 0, 144]
    elif nSubC == 1024:
        nSymHalfSubfr = 14
        maxnRB = 51
        indexTranMat = [718, 1024, 0, 306]
    elif nSubC == 2048:
        nSymHalfSubfr = 7
        maxnRB = 106
        indexTranMat = [1412, 2048, 0, 636]
    
    freqBinSize = nFreqRB * maxnRB
    symLenVec = np.ones((nSym+1, )) * (nSubC + CPLen)
    if CPLen != 128:
        symIndexVec = firstSymIndex + np.arange(nSym+1)
        symLenVec = symLenVec + (np.mod(symIndexVec, nSymHalfSubfr) == 0) * 16
        longSymVec = np.where(np.mod(symIndexVec, nSymHalfSubfr) == 0)[0]
        tempCumulIndex = np.concatenate(([0], np.cumsum(symLenVec)))
        addLongCP = (np.arange(16).reshape(16,1) + tempCumulIndex[longSymVec]).reshape(-1,)
        inputCrop = inputIQ[(firstIndex+np.setdiff1d(np.arange(tempCumulIndex[-1]), addLongCP)).astype(int)]
    else:
        inputCrop = inputIQ[(firstIndex+np.arange(symLenVec.sum())).astype(int)]
    
    for iSym in range(nSym + 1):
        chIndexSym.append(np.mod(getChIndexSym(indexNRDL, nSubC, CPLen, iSym+firstSymIndex)-1, freqBinSize).astype(int))

    featAbs = np.nan * np.zeros((freqBinSize, nSym))
    featPh = np.nan * np.zeros((freqBinSize, nSym))

    prevSymFreq = np.fft.fft(inputCrop[int(CPLen/2):int(CPLen/2)+nSubC])
    prevSymIndex = np.concatenate((prevSymFreq[indexTranMat[0]:indexTranMat[1]],\
        prevSymFreq[indexTranMat[2]:indexTranMat[3]]))
    
    prevSymAbs = np.nan * np.zeros((freqBinSize, ))
    prevSymPh = np.nan * np.zeros((freqBinSize, ))
    
    prevSymAbs[chIndexSym[0]] = np.abs(prevSymIndex[chIndexSym[0]])
    prevSymPh[chIndexSym[0]] = np.angle(prevSymIndex[chIndexSym[0]])

    for iSym in range(1, nSym+1):
        if CPLen != 128 and np.mod(iSym+firstSymIndex, nSymHalfSubfr) == 0:
            continue
        currSymFreq = np.fft.fft(inputCrop[int(CPLen/2)+iSym*(nSubC+CPLen):\
            int(CPLen/2)+iSym*(nSubC+CPLen)+nSubC])
        currSymIndex = np.concatenate((currSymFreq[indexTranMat[0]:indexTranMat[1]],\
            currSymFreq[indexTranMat[2]:indexTranMat[3]]))
        
        currSymAbs = np.nan * np.zeros((freqBinSize, ))
        currSymPh = np.nan * np.zeros((freqBinSize, ))
        
        currSymAbs[chIndexSym[iSym]] = np.abs(currSymIndex[chIndexSym[iSym]])
        currSymPh[chIndexSym[iSym]] = np.angle(currSymIndex[chIndexSym[iSym]])
        
        validFeatIndices = np.intersect1d((np.where(~np.isnan(prevSymAbs))[0]),\
                                          (np.where(~np.isnan(currSymAbs))[0]))
        featAbs[validFeatIndices, iSym-1] = prevSymAbs[validFeatIndices]
        featPh[validFeatIndices, iSym-1] = currSymPh[validFeatIndices] - prevSymPh[validFeatIndices]

        prevSymAbs = currSymAbs
        prevSymPh = currSymPh
    
    if angleMod:
        featPh = np.mod(featPh, np.pi/2)
    feat = np.multiply(featAbs, np.exp(1j*featPh))

    return feat

# plt.figure(figsize=(10,6))
# plt.scatter(np.real(feat), np.imag(feat))
# xylim = 20
# plt.xlim((0, xylim))
# plt.ylim((0, xylim))