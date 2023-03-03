import math
import numpy as np

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

