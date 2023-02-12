import numpy as np

def getMCS_HEMU(inputJson):
    nUser = len(inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'])
    MCS_list = np.zeros((nUser,), dtype=int)
    for iUser in range(nUser):
        MCS_list[iUser] = inputJson['sourceArray'][0]['signalArray']['cfgHEMU']['User'][iUser]['MCS']

    return MCS_list

