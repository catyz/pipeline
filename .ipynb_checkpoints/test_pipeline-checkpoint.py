from toast_pb2 import pb2tod
import toast
import numpy as np

#channel = 'PB20.13.10_Comb30Ch28'
detector = '13.10_112.90B'
tod = pb2tod(datafolder='/global/cscratch1/sd/yzh/data/ChileData/Run20000065/')

pointing = tod.read_pntg(detector) #calls pb2tod._get_pntg()
boresight = tod.read_boresight()

import toast.qarray as qa
theta, phi = qa.to_position(boresight)
az = 2 * np.pi - phi
el = np.pi / 2 - theta
print(min(az),max(az),min(el),max(el))

# import pickle
# pickle.dump(boresight, open('boresight_quats.pkl','wb'))
# pickle.dump(pointing, open('pointing_quats.pkl','wb'))

# print(boresight)
# print(pointing)

# #comm = toast.Comm()
# #data = toast.Data(comm)
