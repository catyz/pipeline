
from spt3g import core, dfmux, gcp
import sys
print(sys.executable)
import numpy as np
from glob import glob

ddir = '/global/cscratch1/sd/yzh/data/ChileData/Run20000065/'
dfiles = sorted(glob(ddir + '*.g3'))

az_buff = []
el_buff = []

def get_azel(frame):
    if frame.type == core.G3FrameType.Scan:
        az_buff.append(frame['TrackerStatus'].az_pos)
        el_buff.append(frame['TrackerStatus'].el_pos)
        
pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=dfiles)
pipe.Add(get_azel)
pipe.Run()

az = np.concatenate(az_buff)
el = np.concatenate(el_buff)
