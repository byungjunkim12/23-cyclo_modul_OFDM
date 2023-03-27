import numpy as np
from scipy.signal import find_peaks
from datetime import datetime
import torch
from torch.utils.data import Dataset


def getCFOtruth(inputJson):
    CFOtruth = inputJson['sourceArray'][0]['imperfectionCfg']['freqOffset']
    return CFOtruth

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
    # print(np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen))
    firstIndex = np.mod(np.bincount(np.mod(peaks, nSubC+CPLen)).argmax(), nSubC+CPLen)

    return firstIndex

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

class IQDataset(Dataset):
    """
    Dataset for spectrogram processing based on torch Dataset class, used with torch DataLoader class to generate
    data batches for training neural network
    """
    def __init__(self,data_dict,cuda_id=None,normalize=True):
        self._features = data_dict['input'] # using IQ sample value
        # self._start = data_dict['start']
        self._labels = data_dict['label']
        self._cuda_id = cuda_id
        self._normalize = normalize
                
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                        
        Input = torch.tensor(self._features[idx]).float()  # adding singleton dimension for CNN
        # Input = torch.tensor(self._features[idx]).unsqueeze(0).float()  # adding singleton dimension for CNN
        Target = torch.tensor(self._labels[idx]).long()
        # start = torch.tensor(self._start[idx]).long()

        if self._normalize == True:
            flattened_input = torch.flatten(Input,start_dim=1)
            spec_mag = torch.sum(torch.norm(flattened_input,dim=0))
            Input = Input * flattened_input.size(dim=1) / spec_mag
            
        if self._cuda_id is not None:
            return {"input": Input.cuda(self._cuda_id), "target": Target.cuda(self._cuda_id)}

        else:
            return {"input": Input, "target": Target}
