import numpy as np
import sqlite3
import pickle
from toast.tod.sim_noise import AnalyticNoise
from toast.weather import Weather

#makes the noise object
def noise():
    fp = load_focalplane()
    detnames = list(sorted(fp.keys()))
    noise_object = AnalyticNoise(
        detectors = detnames,
        rate = {x: fp[x]['rate'] for x in detnames},
        fmin = {x: fp[x]['fmin'] for x in detnames},
        fknee = {x: fp[x]['fknee'] for x in detnames},
        alpha = {x: fp[x]['alpha'] for x in detnames},
        NET = {x: fp[x]['NET'] for x in detnames}
    )
    return noise_object

def weather():
    return Weather('./weather_Atacama.fits')

def load_focalplane():
    return pickle.load(open('focalplane.pkl','rb'))
    
#converts channel name to boloname        
def ch_to_bolo(ch_name):
    conn = sqlite3.connect('boloid_pb2a.db')
    c = conn.cursor()
    data = c.execute("SELECT * from boloid").fetchall()
    ch = np.array(list(d[0] for d in data))
    bolo = np.array(list(d[2] for d in data))
    bolo_name = bolo[np.where(ch_name == ch)]
    return bolo_name[0]  

#converts boloname to channel name
def bolo_to_ch(bolo_name):
    conn = sqlite3.connect('boloid_pb2a.db')
    c = conn.cursor()
    data = c.execute("SELECT * from boloid").fetchall()
    ch = np.array(list(d[0] for d in data))
    bolo = np.array(list(d[2] for d in data))
    ch_name = ch[np.where(bolo_name == bolo)]
    return ch_name[0]  

#gets detector offsets in quats
def detquats():
    fp = pickle.load(open('focalplane.pkl', 'rb'))
    detnames = list(sorted(fp.keys()))
    detquats = {x: fp[x]['quat'] for x in detnames}
    return detquats