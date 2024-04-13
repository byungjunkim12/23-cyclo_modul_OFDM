import math
import numpy as np

def estCFO(inputIQ, nSubC, lenCP, firstIndexSym, samplingRate):
    if nSubC == 512 or nSubC == 1024 or nSubC == 2048:
        if nSubC == 512:
            inputLen = int(0.25e-3 * samplingRate)
        elif nSubC == 1024:
            inputLen = int(0.5e-3 * samplingRate)
        elif nSubC == 2048:
            inputLen = int(1e-3 * samplingRate)
        inputIQ = inputIQ[:inputLen]

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