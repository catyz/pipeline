import sqlite3
import pickle
import numpy as np
from toast.tod.sim_focalplane import cartesian_to_quat

conn = sqlite3.connect('focalplane_pb2a.db')
c = conn.cursor()

data = c.execute('SELECT * from focalplane').fetchall()

dets = np.array(list(d[0] for d in data))
pol_angle = np.array(list(d[3] for d in data))
x_pos = np.array([d[1] for d in data])
y_pos = np.array([d[2] for d in data])

conn.close()

size = len(dets)

#pointing offsets using fred's ray tracing
radius = 180.8 # mm
offset = -2.26 # degrees
conv = offset / radius
offset_x = x_pos * conv 
offset_y = y_pos * conv 

offsets = []
for i in range(len(dets)):
    offsets.append(np.array([offset_x[i], offset_y[i], pol_angle[i]]))

quats = cartesian_to_quat(offsets)
                   

#noise properties, made up for now
fknee = np.ones(size) * 0.1
fmin = np.ones(size) * 1e-05
alpha = np.ones(size) * 1
net = np.ones(size) *  0.0001

fwhm_deg = np.ones(size) * 0.05
fwhm = np.ones(size) * 3
rate = np.ones(size) * 152.6
bandcenter_ghz = np.ones(size) * 150
bandwidth_ghz = np.ones(size) * 10

#fill focal plane dictionary
focalplane = {}
for i in range(size):
    d = {'quat': quats[i],
         'polangle_deg': pol_angle[i],
         'fknee': fknee[i],
         'fmin': fmin[i],
         'alpha': alpha[i],
         'NET': net[i],
         'fwhm_deg': fwhm_deg[i],
         'fwhm': fwhm[i],
         'rate': rate[i],
         'bandcenter_ghz': bandcenter_ghz[i],
         'bandwidth_ghz': bandwidth_ghz[i]
        }
    focalplane[dets[i]] = d

pickle.dump(focalplane, open("focalplane.pkl","wb"))


