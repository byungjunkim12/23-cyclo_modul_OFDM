import numpy as np
from scipy.signal import find_peaks
from datetime import datetime

def getChIndexSym(indexNRDL, nSubC, CPLen, iSym):
    if nSubC == 512 and CPLen == 128:
        freqBinSize = 12*24
        nSymSlot = 12
    elif nSubC == 512 and CPLen == 36:
        freqBinSize = 12*24
        nSymSlot = 14
    elif nSubC == 1024:
        freqBinSize = 12*51
        nSymSlot = 14
    elif nSubC == 2048:
        freqBinSize = 12*106
        nSymSlot = 14
    iSlot = int(np.floor(iSym/nSymSlot))
    
    chIndexSlot = np.union1d(np.union1d(np.union1d(np.union1d(np.union1d(\
        indexNRDL['indexPDSCH'][iSlot],\
        indexNRDL['indexPDSCHDMRS'][iSlot]),\
        indexNRDL['indexPDSCHPTRS'][iSlot]),\
        indexNRDL['indexPDCCH'][iSlot]),\
        indexNRDL['indexSSBurst'][iSlot]),\
        indexNRDL['indexCSIRS'][iSlot])
    chIndexSym = chIndexSlot[(chIndexSlot > np.mod(iSym, nSymSlot) * freqBinSize) &\
        (chIndexSlot <= (np.mod(iSym, nSymSlot)+1) * freqBinSize)]
    return chIndexSym

def getindexNRDL(inputJson, sourceIndex):
    if sourceIndex == 0:
        indexNRDL = inputJson['sourceArray'][9]['signalArray']['indexNRDL']
    else:
        indexNRDL = inputJson['sourceArray'][sourceIndex-1]['signalArray']['indexNRDL']
    # indexNRDL = inputJson['sourceArray'][sourceIndex]['signalArray']['indexNRDL']
    return indexNRDL

def getCFOtruth(inputJson):
    CFOtruth = inputJson['sourceArray'][0]['imperfectionCfg']['freqOffset']
    return CFOtruth

def getMCS_HEMU(inputJson):
    nUser = len(inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'])
    MCS_list = np.zeros((nUser,), dtype=int)
    for iUser in range(nUser):
        MCS_list[iUser] = inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'][iUser]['MCS']

    return MCS_list

def findFirstIndexWifi(inputIQ, nSubC, CPLen):
    indexMat = np.arange(CPLen)[:, np.newaxis] + np.arange(inputIQ.shape[0]-CPLen+1)
    inputMat = inputIQ[indexMat]
    
    inputCorr = np.sum(inputMat[:, :-nSubC] * np.conj(inputMat[:, nSubC:]), axis=0)
    peaks, pklocs = find_peaks(np.abs(inputCorr), distance=np.floor(0.9*(nSubC+CPLen)))
    # print(np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen))
    firstIndex = np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen)

    return firstIndex

def findTrueFirstIndexNR(inputStartIndex, nSubC, CPLen):
    inputStartIndexSubfr = inputStartIndex % 15360
    if CPLen == 128:
        firstIndexTruth = (-inputStartIndexSubfr) % (nSubC+CPLen)
        # firstSymIndexTruth = None
        firstSymIndexTruth = np.ceil(inputStartIndex / (nSubC+CPLen)).astype(int)
    else:
        if inputStartIndexSubfr > 15360 - (nSubC+CPLen):
            firstSymIndexTruth = 0
            if inputStartIndexSubfr <= 15360 - (nSubC+CPLen-16):
                firstIndexTruth = (-inputStartIndexSubfr+32) % (nSubC+CPLen) + (nSubC+CPLen)
            else:
                firstIndexTruth = (-inputStartIndexSubfr+32) % (nSubC+CPLen)
        else:
            firstIndexTruth = (-inputStartIndexSubfr + 16) % (nSubC+CPLen)
            firstSymIndexTruth = np.ceil((inputStartIndexSubfr-16) / (nSubC+CPLen)).astype(int)

    return firstIndexTruth, firstSymIndexTruth


def findFirstIndexNR(inputIQ, nSubC, CPLen):
    dataSamplingRate = 30.72e6
    lenHalfSubfr = int(0.5e-3 * dataSamplingRate)

    nHalfSubfrInput = np.floor(inputIQ.shape[0]/lenHalfSubfr).astype(int)
    if nSubC == 512 and CPLen == 128:
        nSymSlot = 48
    elif nSubC == 512 and CPLen == 36:
        nSymSlot = 56
    elif nSubC == 1024:
        nSymSlot = 28
    elif nSubC == 2048:
        nSymSlot = 14
    nSymInput = int(nSymSlot * nHalfSubfrInput / 2)
    addInputLen = np.mod(inputIQ.shape[0], lenHalfSubfr)
    # print(nSymInput, nHalfSlotInput, addInputLen)
    
    indexMat = np.arange(CPLen)[:, np.newaxis] + np.arange(inputIQ.shape[0]-CPLen+1)
    inputMat = inputIQ[indexMat]
    inputCorr = np.sum(inputMat[:, :-nSubC] * np.conj(inputMat[:, nSubC:]), axis=0)

    if CPLen == 128:
        peaks = find_peaks(np.abs(inputCorr), distance=np.floor(0.9*(nSubC+CPLen)))[0]
        firstIndex = np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen)
        firstSymIndex = None
    else:
        pkMat2Flag = False

        pkMat = np.nan*np.zeros((int(nSymInput/nHalfSubfrInput)+2, nHalfSubfrInput))
        for iSlot in range(nHalfSubfrInput):
            inputCorrTemp = inputCorr[iSlot*lenHalfSubfr : (iSlot+1)*lenHalfSubfr+addInputLen]
            peaks = find_peaks(np.abs(inputCorrTemp), distance=np.floor(0.9*(nSubC+CPLen)))[0]
            # to find peaks whose distance is larger than 90% of the length of a symbol length
            pkSym = np.floor(peaks / (nSubC+CPLen)).astype(int)
            pkValidIndices = np.where((pkSym <= int(nSymInput/nHalfSubfrInput)+2) & (pkSym > 0))[0]
            # to remove peaks that are in the beginning or the end of the input
            pkMat[pkSym[pkValidIndices]-1, iSlot] = peaks[pkValidIndices]
            # print('peaks:', peaks, 'pkMat:', pkMat[:, iSlot])

            nanIndex = np.setdiff1d(np.where(np.isnan(pkMat[:, iSlot]))[0], [pkMat.shape[0]-1])
            nanIndex = np.where(np.isnan(pkMat[:, iSlot]))[0]
            # print('nan:', nanIndex)
            pkMat[np.setdiff1d(nanIndex, np.array([int(nSymInput/nHalfSubfrInput)+1]))+1, iSlot] = np.nan;
            pkMat[np.setdiff1d(nanIndex, np.array([0, int(nSymInput/nHalfSubfrInput)+1]))-1, iSlot] = np.nan;
        
        pkMat2 = np.nan*np.zeros((int(nSymInput/nHalfSubfrInput)+2, nHalfSubfrInput))
        for iSlot in range(nHalfSubfrInput):
            inputCorrTemp = inputCorr[iSlot*lenHalfSubfr+int((nSubC+CPLen)/2) :\
                                      (iSlot+1)*lenHalfSubfr+addInputLen+int((nSubC+CPLen)/2)]
            peaks = find_peaks(np.abs(inputCorrTemp), distance=np.floor(0.9*(nSubC+CPLen)))[0]
            # to find peaks whose distance is larger than 90% of the length of a symbol length
            pkSym = np.floor(peaks / (nSubC+CPLen)).astype(int)
            pkValidIndices = np.where((pkSym <= int(nSymInput/nHalfSubfrInput)+2) & (pkSym > 0))[0]
            # to remove peaks that are in the beginning or the end of the input
            pkMat2[pkSym[pkValidIndices]-1, iSlot] = peaks[pkValidIndices]
            # print('peaks:', peaks, 'pkMat:', pkMat2[:, iSlot])

            nanIndex = np.setdiff1d(np.where(np.isnan(pkMat2[:, iSlot]))[0], [pkMat2.shape[0]-1])
            nanIndex = np.where(np.isnan(pkMat2[:, iSlot]))[0]
            # print('nan:', nanIndex)
            pkMat2[np.setdiff1d(nanIndex, np.array([int(nSymInput/nHalfSubfrInput)+1]))+1, iSlot] = np.nan;
            pkMat2[np.setdiff1d(nanIndex, np.array([0, int(nSymInput/nHalfSubfrInput)+1]))-1, iSlot] = np.nan;

        # print(pkMat)
        # print(pkMat2)

        if np.sum(np.isnan(pkMat)) > np.sum(np.isnan(pkMat2)):
            # print('changed')
            pkMat = pkMat2
            pkMat2Flag = True

        pkRem = np.mod(pkMat, nSubC+CPLen)
        pkRemAvg = np.nanmean(pkRem, axis=1)
        symOffset = np.floor(np.nanmean(pkMat[0, :]) / (nSubC+CPLen)).astype(int)

        remDiff = np.abs(pkRemAvg[2:] - pkRemAvg[:-2])
        symIndexCandi = np.argsort(remDiff)[-2:]
        if (remDiff[symIndexCandi[0]] > 10 and remDiff[symIndexCandi[1]] > 10):
            minStdIndex = np.argmax(np.nanstd(pkRem[symIndexCandi+1, :], axis=1))
            firstSymIndex = symIndexCandi[minStdIndex]
        else:
            firstSymIndex = symIndexCandi[1]
        firstSymIndex = np.mod(-(firstSymIndex+symOffset+1), nSymSlot/2).astype(int)
        
        symVec = np.arange(pkRemAvg.size)
        shortSymVec = symVec[np.mod(symVec+firstSymIndex, nSymSlot/2) != 0]
        
        pkRemAvg = pkRemAvg - np.floor((symVec+firstSymIndex+symOffset) / (nSymSlot/2)) * 16
        pkRemAvg = np.mod(pkRemAvg[shortSymVec], nSubC+CPLen)
        # print(pkRemAvg.astype(int), np.bincount(pkRemAvg.astype(int))[896], np.bincount(pkRemAvg.astype(int))[1078])
        if np.unique(pkRemAvg).size == np.size(pkRemAvg):
            firstIndex = np.nanmedian(pkRemAvg).astype(int)
        else:
            firstIndex = np.bincount(pkRemAvg.astype(int)).argmax()
        
        if pkMat2Flag:
            # print(firstIndex, firstSymIndex)
            if firstSymIndex == 0 and firstIndex >= (nSubC+CPLen)/2 and firstIndex < (nSubC+CPLen)/2+16:
                firstIndex = (firstIndex + (nSubC+CPLen)/2).astype(int)
            elif firstIndex >= (nSubC+CPLen)/2:
                firstIndex = (firstIndex - (nSubC+CPLen)/2).astype(int)
                firstSymIndex = np.mod(firstSymIndex-1, nSymSlot/2).astype(int)
            else:
                firstIndex = (firstIndex + (nSubC+CPLen)/2).astype(int)

    return [firstIndex, firstSymIndex]

def getDateYYMMDD_HHMMSS():
    dateTimeNow = datetime.now()
    dateYY = getTwoDigitStr(dateTimeNow.year % 100)
    dateMM = getTwoDigitStr(dateTimeNow.month)
    dateDD = getTwoDigitStr(dateTimeNow.day)
    dateHH = getTwoDigitStr(dateTimeNow.hour)
    dateMi = getTwoDigitStr(dateTimeNow.minute)
    dateSe = getTwoDigitStr(dateTimeNow.second)

    DateYYMMDD_HHMMSS = dateYY + dateMM + dateDD + "_" + dateHH + dateMi + dateSe
    return DateYYMMDD_HHMMSS

def getTwoDigitStr(input):
    if input < 10:
        twoDigitStr = "0" + str(input)
    else:
        twoDigitStr = str(input)
    return twoDigitStr

