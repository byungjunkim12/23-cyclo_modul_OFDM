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

    def __init__(self,data_dict,imgSize,angleMod,cuda_id=None,normalize=True):
        self._features = data_dict['input'] # using IQ sample value
        self._imgSize = imgSize
        self._labels = data_dict['label']
        self._cuda_id = cuda_id
        self._normalize = normalize
        self._angleMod = angleMod
        
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                        
        Input = torch.tensor(self._features[idx]).float()  # adding singleton dimension for CNN
        Target = torch.tensor(self._labels[idx]).long()
        # Input = torch.divide(Input, torch.numel(Input))

        if self._cuda_id is not None:
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

class wifi16QamNet(nn.Module):
    def __init__(self, nLabel, removeNull):
        super().__init__()
        self.nLabel = nLabel
        self.input_fc = 928

        self.conv1 = nn.Conv2d(2, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.conv3 = nn.Conv2d(8, 16, 2)
        self.conv4 = nn.Conv2d(16, 32, 2)
        if removeNull:
            self.input_fc = 12288
        else:
            self.input_fc = 448

        self.fc1 = nn.Linear(self.input_fc, 2400)
        self.fc2 = nn.Linear(2400, 84)
        self.fc3 = nn.Linear(84, self.nLabel)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # print(x.shape[0])
        if x.shape[0] == 1:
            x = torch.unsqueeze(torch.squeeze(self.pool(F.relu(self.conv1(x)))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv2(x))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv3(x))), 0)
            x = torch.unsqueeze(torch.squeeze(F.relu(self.conv4(x))), 0)
        else:
            x = torch.squeeze(self.pool(F.relu(self.conv1(x))))
            x = torch.squeeze(F.relu(self.conv2(x)))
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


def getPredsWifi(loader,modelPsk,model16Qam,print_time = False, snr_key = 'EsNo'):
    '''
    get predictions from network
    '''

    model16Qam.eval()
    modelPsk.eval()
    pred_l   = []
    target_l = [] 

    start = time.time()
        
    for batch in loader:
        batchInputComplex = torch.squeeze(batch['input'][:, 0, :, :] + 1j * batch['input'][:, 1, :, :])
        batchInputAbs = torch.abs(batchInputComplex)

        batchInputPh =  torch.remainder(torch.angle(batchInputComplex), (pi/2))
        # batchInputPh[batchInputPh > pi/4] = pi/2 - batchInputPh[batchInputPh > pi/4]
        batchInputMod = torch.multiply(batchInputAbs, torch.exp(1j*batchInputPh))
        batchInputQam = torch.cat((torch.unsqueeze(torch.real(batchInputMod), 1), \
            torch.unsqueeze(torch.imag(batchInputMod), 1)), 1)

        outputsQam = model16Qam(batchInputQam).detach().max(dim=1).indices.cpu()
        outputsPsk = modelPsk(batch['input']).detach().max(dim=1).indices.cpu()
        outputsPsk[outputsPsk == 2] = outputsQam[outputsPsk == 2] + 2
        # outputsQam[outputsQam < 2] = outputsPsk[outputsQam < 2]

        # pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        pred_l.extend(outputsPsk.tolist())
        target_l.extend(batch['target'].cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(target_l))

    return pred_l, target_l


def getAccWifi(loader, modelPsk, model16Qam):
    '''
    get accuracy from predictions
    '''
    # pred_l,target_l,_ = get_preds(loader, model)
    pred_l,target_l = getPredsWifi(loader, modelPsk, model16Qam)

    correct = 0.
    for pred, target in zip(pred_l,target_l):
        correct += pred==target

    return correct/len(pred_l)
