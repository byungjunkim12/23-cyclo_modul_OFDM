from cmath import pi
import numpy as np
import math

pi = math.pi
def getModFeat(inputIQ, nSubC, lenCP, firstIndexSym, nSym, removeNull, angleMod):
    featAbs = np.zeros((nSubC, nSym))
    featPh = np.zeros((nSubC, nSym))
    if nSubC == 64:
        nNullSubC = 8
    elif nSubC == 256:
        nNullSubC = 32

    for iSym in range(nSym):
        symbolsFreq = np.fft.fft(inputIQ[(firstIndexSym+int(lenCP/2) + iSym*(nSubC+lenCP)) :\
            (firstIndexSym+nSubC+int(lenCP/2) + iSym*(nSubC+lenCP))])
        nextSymbolsFreq = np.fft.fft(inputIQ[(firstIndexSym+int(lenCP/2) + (iSym+1)*(nSubC+lenCP)) :\
            (firstIndexSym+nSubC+int(lenCP/2) + (iSym+1)*(nSubC+lenCP))])

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
    
# plt.figure(figsize=(10,6))
# plt.scatter(np.real(feat), np.imag(feat))
# xylim = 20
# plt.xlim((0, xylim))
# plt.ylim((0, xylim))