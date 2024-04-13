import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
import math

pi = math.pi
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

class sphDataset(Dataset):
    """
    Dataset for spectrogram processing based on torch Dataset class, used with torch DataLoader class to generate
    data batches for training neural network
    """

    def __init__(self,data_dict,cuda_id=None,normalize=True):
        self._features = data_dict['input'] # using IQ sample value
        self._labels = data_dict['label']
        self._cuda_id = cuda_id
        self._normalize = normalize
        
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                        
        Input = torch.tensor(self._features[idx]).float()  # adding singleton dimension for CNN
        Target = torch.tensor(self._labels[idx]).long()

        if self._normalize == True:
            Input[0, :, :] = Input[0, :, :] / torch.mean(Input[0, :, :])
            Input[1, :, :] = Input[1, :, :] / math.pi
            
        if self._cuda_id is not None:
            return {"input": Input.cuda(self._cuda_id), "target": Target.cuda(self._cuda_id)}

        else:
            return {"input": Input, "target": Target}

class imgDataset(Dataset):
    """
    Dataset for spectrogram processing based on torch Dataset class, used with torch DataLoader class to generate
    data batches for training neural network
    """

    def __init__(self,data_dict,testFlag,cuda_id=None,normalize=True):
        self._features = data_dict['input'] # using IQ sample value
        # self._imgSize = imgSize
        self._labels = data_dict['label']
        self._cuda_id = cuda_id
        self._normalize = normalize
        self._testFlag = testFlag
        if self._testFlag:
            self._featuresMod = data_dict['inputMod']
        
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                        
        Input = torch.tensor(self._features[idx]).float()  # adding singleton dimension for CNN
        Target = torch.tensor(self._labels[idx]).long()
        if self._testFlag:
            InputMod = torch.tensor(self._featuresMod[idx]).float()
        # Input = torch.divide(Input, torch.numel(Input))

        if self._cuda_id is not None:
            if self._testFlag:
                return {"input": Input.cuda(self._cuda_id), "target": Target.cuda(self._cuda_id), "inputMod": InputMod.cuda(self._cuda_id)}
            else:
                return {"input": Input.cuda(self._cuda_id), "target": Target.cuda(self._cuda_id)}
        else:
            return {"input": Input, "target": Target}




def IQData(audio_input):
    inputReal = np.expand_dims(np.real(audio_input), axis=0)
    inputImag = np.expand_dims(np.imag(audio_input), axis=0)

    IQ = np.concatenate((inputReal, inputImag), axis=0)
    return IQ

def sphData(audio_input):
    inputMag = np.absolute(audio_input)
    inputPh = np.angle(audio_input)

    inputMag = np.expand_dims(inputMag, axis=0)
    inputPh = np.expand_dims(inputPh, axis=0)

    sph = np.concatenate((inputMag, inputPh), axis=0)
    return sph

def imgData(audio_input, imgSize, angleMod):
    horIndex = np.ceil(np.abs(audio_input) / (np.nanmax(np.abs(audio_input)) / imgSize)).reshape(-1, 1)
    if angleMod:
        verIndex = np.ceil(np.angle(audio_input) / ((pi/2) / imgSize)).reshape(-1, 1)
    else:
        verIndex = np.ceil(np.angle(audio_input) / (pi / imgSize)).reshape(-1, 1)
    horIndex[horIndex == imgSize] = imgSize-1
    verIndex[verIndex == imgSize] = imgSize-1
        
    horIndexIndicator = np.reshape(np.arange(imgSize), (-1, 1)) ==\
        np.reshape(horIndex, (1, 1, np.size(horIndex)))
    if angleMod:
        verIndexIndicator = np.transpose(np.reshape(np.arange(imgSize), (-1, 1)) ==\
            (np.reshape(verIndex, (1, 1, np.size(verIndex)))), (1,0,2))
    else:
        verIndexIndicator = np.transpose(np.reshape(np.arange((-imgSize+1), (imgSize+1), 2), (-1, 1)) ==\
            (np.reshape(verIndex, (1, 1, np.size(verIndex)))), (1,0,2))
    # print(horIndexIndicator.shape, verIndexIndicator.shape)

    Input = np.expand_dims(np.sum(horIndexIndicator * verIndexIndicator, axis=2).astype(float), 0)
    # print(Input.shape)
    return Input
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)





class wifiPskNet(nn.Module):
    def __init__(self, nLabel, removeNull):
        super().__init__()
        self.nLabel = nLabel
        self.input_fc = 928

        self.conv1 = nn.Conv2d(2, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.conv3 = nn.Conv2d(8, 16, 2)
        self.conv4 = nn.Conv2d(16, 32, 2)
        if removeNull:
            self.input_fc = 2464
            # self.input_fc = 288
        else:
            self.input_fc = 448

        self.fc1 = nn.Linear(self.input_fc, 1200)
        self.fc2 = nn.Linear(1200, 84)
        self.fc3 = nn.Linear(84, self.nLabel)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # print(x.shape[0])
        if x.shape[0] == 1:
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv1(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv2(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv3(x))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv4(x))), 0)
        else:
            x = torch.squeeze(self.pool(F.relu(self.conv1(x))))
            x = torch.squeeze(self.pool(F.relu(self.conv2(x))))
            x = torch.squeeze(F.relu(self.conv3(x)))
            x = torch.squeeze(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class wifiImgPskNet(nn.Module):
    def __init__(self, nLabel, removeNull):
        super().__init__()
        self.nLabel = nLabel
        self.input_fc = 928

        self.conv1 = nn.Conv2d(1, 2, 2)
        self.conv2 = nn.Conv2d(2, 4, 2)
        self.conv3 = nn.Conv2d(4, 8, 2)
        self.conv4 = nn.Conv2d(8, 16, 2)
        if removeNull:
            # self.input_fc = 7744
            # self.input_fc = 144
            self.input_fc = 16
        else:
            self.input_fc = 448

        self.fc1 = nn.Linear(self.input_fc, 1200)
        self.fc2 = nn.Linear(1200, 84)
        self.fc3 = nn.Linear(84, self.nLabel)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # print(x.shape[0])
        if x.shape[0] == 1:
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv1(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv2(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv3(x))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv4(x))), 0)
        else:
            x = torch.squeeze(self.pool(F.relu(self.conv1(x))))
            x = torch.squeeze(self.pool(F.relu(self.conv2(x))))
            x = torch.squeeze(F.relu(self.conv3(x)))
            x = torch.squeeze(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class wifiImgQamNet(nn.Module):
    def __init__(self, nLabel, removeNull):
        super().__init__()
        self.nLabel = nLabel
        self.input_fc = 928

        self.conv1 = nn.Conv2d(1, 2, 2)
        self.conv2 = nn.Conv2d(2, 4, 2)
        self.conv3 = nn.Conv2d(4, 8, 2)
        self.conv4 = nn.Conv2d(8, 16, 2)
        if removeNull:
            self.input_fc = 256
            # self.input_fc = 1296
        else:
            self.input_fc = 448

        self.fc1 = nn.Linear(self.input_fc, 1200)
        self.fc2 = nn.Linear(1200, 84)
        self.fc3 = nn.Linear(84, self.nLabel)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # print(x.shape[0])
        if x.shape[0] == 1:
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv1(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv2(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv3(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv4(x))), 0)
        else:
            x = torch.squeeze(self.pool(F.relu(self.conv1(x))))
            x = torch.squeeze(self.pool(F.relu(self.conv2(x))))
            x = torch.squeeze(self.pool(F.relu(self.conv3(x))))
            x = torch.squeeze(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NRImgNet(nn.Module):
    def __init__(self, nLabel, removeNull):
        super().__init__()
        self.nLabel = nLabel
        self.input_fc = 928

        self.conv1 = nn.Conv2d(1, 2, 2)
        self.conv2 = nn.Conv2d(2, 4, 2)
        self.conv3 = nn.Conv2d(4, 8, 2)
        self.conv4 = nn.Conv2d(8, 16, 2)
        if removeNull:
            self.input_fc = 1600
            # self.input_fc = 1296
        else:
            self.input_fc = 448

        self.fc1 = nn.Linear(self.input_fc, 1200)
        self.fc2 = nn.Linear(1200, 84)
        self.fc3 = nn.Linear(84, self.nLabel)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # print(x.shape[0])
        if x.shape[0] == 1:
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv1(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv2(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv3(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv4(x))), 0)
        else:
            x = torch.squeeze(self.pool(F.relu(self.conv1(x))))
            x = torch.squeeze(self.pool(F.relu(self.conv2(x))))
            x = torch.squeeze(self.pool(F.relu(self.conv3(x))))
            x = torch.squeeze(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class wifiSeqNet(nn.Module):
    def __init__(self, nLabel, removeNull):
        super().__init__()
        self.nLabel = nLabel
        self.input_fc = 928

        self.conv1 = nn.Conv2d(2, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.conv3 = nn.Conv2d(8, 16, 2)
        self.conv4 = nn.Conv2d(16, 32, 2)
        if removeNull:
            self.input_fc = 2464
            # self.input_fc = 288
        else:
            self.input_fc = 448

        self.fc1 = nn.Linear(self.input_fc, 1200)
        self.fc2 = nn.Linear(1200, 84)
        self.fc3 = nn.Linear(84, self.nLabel)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # print(x.shape[0])
        if x.shape[0] == 1:
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv1(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv2(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv3(x))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv4(x))), 0)
        else:
            x = torch.squeeze(self.pool(F.relu(self.conv1(x))))
            x = torch.squeeze(self.pool(F.relu(self.conv2(x))))
            x = torch.squeeze(F.relu(self.conv3(x)))
            x = torch.squeeze(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def getAcc(loader, model):
    '''
    get accuracy from predictions
    '''
    # pred_l,target_l,_ = get_preds(loader, model)
    pred_l,target_l = getPreds(loader, model)

    correct = 0.
    for pred, target in zip(pred_l,target_l):
        correct += (pred==target)

    return correct/len(pred_l)

def getPreds(loader,model,print_time = False, snr_key = 'EsNo'):
    '''
    get predictions from network
    '''

    model.eval()
    pred_l   = []
    target_l = [] 

    start = time.time()
        
    for batch in loader:
        outputs = model(batch['input'])
        # print('outputs:',outputs.shape)
        pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        target_l.extend(batch['target'].cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(target_l))

    return pred_l, target_l


def getPredsWifi(loader,modelPsk,model16Qam,model64Qam=None,model256Qam=None,print_time = False, snr_key = 'EsNo'):
    '''
    get predictions from network
    '''

    modelPsk.eval()
    model16Qam.eval()
    if model64Qam is not None:
        model64Qam.eval()
    if model256Qam is not None:
        model256Qam.eval()
    pred_l   = []
    target_l = [] 

    start = time.time()
        
    for batch in loader:
        outputsPsk = modelPsk(batch['input']).detach().max(dim=1).indices.cpu()
        outputs16Qam = model16Qam(batch['inputMod']).detach().max(dim=1).indices.cpu()
        outputsPsk[outputsPsk == 2] = outputs16Qam[outputsPsk == 2] + 2

        if model64Qam is not None:
            outputs64Qam = model64Qam(batch['inputMod']).detach().max(dim=1).indices.cpu()
            outputsPsk[outputsPsk == 3] = outputs64Qam[outputsPsk == 3] + 3
        if model256Qam is not None:
            outputs256Qam = model256Qam(batch['inputMod']).detach().max(dim=1).indices.cpu()
            outputsPsk[outputsPsk == 4] = outputs256Qam[outputsPsk == 4] + 4

        pred_l.extend(outputsPsk.tolist())
        target_l.extend(batch['target'].cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(target_l))

    return pred_l, target_l


def getAccWifi(loader, modelPsk, model16Qam, model64Qam=None, model256Qam=None):
    '''
    get accuracy from predictions
    '''
    # pred_l,target_l,_ = get_preds(loader, model)
    pred_l,target_l = getPredsWifi(loader, modelPsk, model16Qam, model64Qam, model256Qam)

    correct = 0.
    for pred, target in zip(pred_l,target_l):
        correct += pred==target

    return correct/len(pred_l)
