from toast.tod import TOD
from glob import glob
from spt3g import core, dfmux, gcp
import numpy as np
import os
from other_tools import *
import pickle
from toast.todmap import TODGround
import ephem
from toast.healpix import ang2vec
import toast.qarray as qa

class pb2tod(TOD):
    
    def __init__(
            self,
            datafolder,
    ):
        self.datafiles = sorted(glob(datafolder+'Run20000065_000.g3'))
        self.name = 'test'
        self.ces_id = 1
        print('number of files: %i' % len(self.datafiles))
        #get noise properties
        self.noise = noise()
        
        print('loading in g3 files')
        f = []
        for datafile in self.datafiles:
            f.append(list(core.G3File(datafile)))
            
        #detectors = []
        #for det in f[0][2]["RawTimestreams_I"]:
            #detectors.append(str(det).split(',')[0][1:]) 
            
        #self.channels = ['PB20.13.12_Comb16Ch10']
        self.channels = ['PB20.13.10_Comb14Ch29']
        #self.channels = ['PB20.13.10_Comb30Ch28']
        
        #since the g3 files use channel names, but everything else uses bolonames, convert channel to bolo for convenience
        detectors = []
        for ch in self.channels:
            detectors.append(ch_to_bolo(ch))
        
        print('peek at total number of samples')
        self.nsamp = 0
        for file in f:
            for frame in file:
                if frame.type == core.G3FrameType.Scan:
                    self.nsamp += len(frame['RawTimestreams_I'][self.channels[0]])
        
        
        print('total samples: {}'.format(self.nsamp))
        
#         props = {
#             "site_lon": -67.786222,
#             "site_lat": -22.958064,
#             "site_alt": 5200
#             "azmin": min(self._az),
#             "azmax": max(self._az),
#             "el": np.median(self._el),
#             "scanrate": self._scan_rate,
#             "scan_accel": self._scan_accel,
#         }
        
        super().__init__(
            mpicomm=None,
            detectors=detectors,
            samples=self.nsamp,
            detindx={x[0]:x[1] for x in zip(detectors, range(len(detectors)))},
            detranks=1,
 #           meta=props
        )
        
        self.load_g3()

    def load_g3(self): 
        #gets the data using the example on bolowiki. Replace with kevin's new way later
        
        ch = self.channels[0]
        bolo = ch_to_bolo(ch)
        
        tod_buff = []
        az_buff = []
        el_buff = []
        bolotime_buff = []
        antennatime_buff = []
        
        def get_TOD(frame):
            if frame.type == core.G3FrameType.Scan:
                tod_buff.append(frame['RawTimestreams_I'][ch])
        
        def get_azel(frame):
            if frame.type == core.G3FrameType.Scan:
                az_buff.append(frame['TrackerStatus'].az_pos)
                el_buff.append(frame['TrackerStatus'].el_pos)
        
        def get_bolotime(frame):
            if frame.type == core.G3FrameType.Scan:
                bolotime_buff.append(np.array([t.time*1e-8 for t in frame['DetectorSampleTimes']]))
                
        def get_antennatime(frame):
            if frame.type == core.G3FrameType.Scan:
                antennatime_buff.append([frame['TrackerStatus'].time[k].time * 1e-8 for k in range(len(frame['TrackerStatus'].time
                ))])
        
        print('getting data from g3')
        pipe = core.G3Pipeline()
        pipe.Add(core.G3Reader, filename=self.datafiles)
        pipe.Add(get_TOD)
        pipe.Add(get_azel)
        pipe.Add(get_bolotime)
        pipe.Add(get_antennatime)
        pipe.Run()
        
        #self.cache.create(name, np.float64, (self.local_samples[1],))
        self._tod = np.concatenate(tod_buff)
        self._az = np.concatenate(az_buff) 
        self._el = np.concatenate(el_buff) 
        self._bolotime = np.concatenate(bolotime_buff)
        self._antennatime = np.concatenate(antennatime_buff)
        
        
        #set the parameters for coordinate conversion later
        self._firsttime = self._antennatime[0]
        self._lasttime = self._antennatime[-1]
        
#         self._antennatime_fixed = np.linspace(self._firsttime, self._lasttime, len(self._antennatime))
#         self._az = np.interp(self._antennatime_fixed, self._antennatime, az)
#         self._el = np.interp(self._antennatime_fixed, self._antennatime, el)
        
        self._site_lon = '-67.786222'
        self._site_lat = '-22.958064'
        self._site_alt = 5200
        
        
        self._observer = ephem.Observer()
        self._observer.lon = self._site_lon
        self._observer.lat = self._site_lat
        self._observer.elevation = self._site_alt
        #self._observer.epoch = ephem.J2000
        #self._observer.compute_pressure()
        
        
#         print('translating pointing')
        #interpolate boresight to tod length
#         self._az = np.interp(self._bolotime, self._antennatime, az)
#         self._el = np.interp(self._bolotime, self._antennatime, el)
        
#         print('writing stuff to file')
#         pickle.dump([self._az, self._el], open('azel_interp.pkl','wb'))
        pickle.dump([self._az, self._el], open('azel.pkl','wb'))
        pickle.dump(self._bolotime, open('bolotime.pkl','wb'))
        pickle.dump(self._antennatime, open('antennatime.pkl','wb'))

        print("bolotime length: %i" % len(self._bolotime))
        print("antennatime length: %i" % len(self._antennatime))
        print("az length: %i" % len(self._az))
        print("tod length: %i" % len(self._tod))
        
#         theta, phi, pa = qa.to_angles(self._boresight_azel)
#         pickle.dump([theta, phi, pa], open('boresight.pkl','wb'))
        
#         print('filling caches')
#         self.cache.put('boresight_quats', self._boresight_azel, replace=False)

        #put stuff in cache
        name = "{}_{}".format(self.SIGNAL_NAME, bolo)
        self.cache.put(name, self._tod, replace=False)
        self.cache.put('timestamps', self._bolotime, replace=False)
        #flag = 0 until we get real stuff
        flag = np.zeros(self.nsamp, dtype=np.uint8)
        name = "{}_{}".format(self.FLAG_NAME, bolo)
        self.cache.put(name, flag, replace=False)
        self.cache.put('common_flags', flag, replace=False)
        
        self.translate_pointing()
        self.cache.put('boresight_quats', self._boresight, replace=False)
        
        #writing stuff for debugging
        theta, phi, pa = qa.to_angles(self._boresight)
        pickle.dump([theta, phi, pa], open('boresight.pkl','wb'))
        
        print("length of boresight vector is: %i" % len(self._boresight))
        return            
    
    def translate_pointing(self):
        """Translate Az/El into bore sight quaternions
        Translate the azimuth and elevation into bore sight quaternions.
        """
        nsamp = len(self._az)
        rank = 0
        ntask = 1
        if self._mpicomm is not None:
            rank = self._mpicomm.rank
            ntask = self._mpicomm.size
        nsamp_task = nsamp // ntask + 1
        my_start = rank * nsamp_task
        my_stop = min(my_start + nsamp_task, nsamp)
        my_nsamp = max(0, my_stop - my_start)
        my_ind = slice(my_start, my_stop)

        my_azelquats = qa.from_angles(
            np.pi / 2 - np.ones(my_nsamp) * self._el,
            -(self._az[my_ind]),
            np.zeros(my_nsamp),
            IAU=False,
        )
        azelquats = None
        if self._mpicomm is None:
            azelquats = my_azelquats
        else:
            azelquats = np.vstack(self._mpicomm.allgather(my_azelquats))
        
#        self._boresight = qa.slerp(self._bolotime, self._antennatime, azelquats)
    
#         pickle.dump(self._boresight_azel, open('boresight_quats_azel.pkl','wb'))
#         my_times = self.local_times()[my_ind]
        
        my_times = self._antennatime#[my_ind]
        azel2radec_times, azel2radec_quats = self._get_azel2radec_quats()
        
#         my_azel2radec_quats = azel2radec_quats
        
#         pickle.dump(azel2radec_quats, open('azel2radec_quats.pkl','wb'))
        my_azel2radec_quats = qa.slerp(my_times, azel2radec_times, azel2radec_quats)
#         print('going into the slerp, my_times is length %i ' % len(my_times))
#         print('going into the slerp, azel2radec_times is length %i ' % len(azel2radec_times))
#         print('going into the slerp, azel2radec_quats is length %i ' % len(azel2radec_quats))
#         my_azelquats_slerped = qa.slerp(my_times, self._antennatime, my_azelquats)
        
#         print('going into the qa.mult, my_azel2radec_quats (which came out of the slerp) is {}'.format(my_azel2radec_quats.shape))
#         print('going into the qa.mult, my_azelquats is {}'.format(my_azelquats.shape))
        my_quats = qa.mult(my_azel2radec_quats, my_azelquats)
        
        #ok so now we have boresight quats translated into radec coordinates. Now we need one last slerp to make it the same length as the timestream
#         print(np.allclose(self._bolotime, self.local_times()))
#         pickle.dump(self._bolotime, open('bolotime.pkl','wb'))
#         pickle.dump(self.local_times(), open('local_times.pkl','wb'))
        
        my_quats_slerped = qa.slerp(self._bolotime, my_times, my_quats)
        
#         print('length of my_quats: %i (should be same as timestream)' % len(my_quats))
        del my_azelquats

        quats = None
        if self._mpicomm is None:
            quats = my_quats_slerped
        else:
            quats = np.vstack(self._mpicomm.allgather(my_quats_slerped))
        self._boresight = quats
        del my_quats
        return


    def _get_azel2radec_quats(self):
        """Construct a sparsely sampled vector of Az/El->Ra/Dec quaternions.
        The interpolation times must be tied to the total observation so
        that the results do not change when data is distributed in time
        domain.
        """
#         One control point at least every 10 minutes.  Overkill but
#         costs nothing.
        n = max(2, 1 + int((self._lasttime - self._firsttime) / 600))
        times = np.linspace(self._firsttime, self._lasttime, n)
        
#         times = self._antennatime
#         n = len(times)
        quats = np.zeros([n, 4])
        for i, t in enumerate(times):
            quats[i] = self._get_coord_quat(t)
            # Make sure we have a consistent branch in the quaternions.
            # Otherwise we'll get interpolation issues.
            if i > 0 and (
                np.sum(np.abs(quats[i - 1] + quats[i]))
                < np.sum(np.abs(quats[i - 1] - quats[i]))
            ):
                quats[i] *= -1
        quats = qa.norm(quats)
        return times, quats

    def _get_coord_quat(self, t):
        """Get the Az/El -> Ra/Dec conversion quaternion for boresight.
        We will apply atmospheric refraction and stellar aberration in
        the detector frame.
        """
        self._observer.date = self.to_DJD(t)
        # Set pressure to zero to disable atmospheric refraction.
        pressure = self._observer.pressure
        self._observer.pressure = 0
        # Rotate the X, Y and Z axes from horizontal to equatorial frame.
        # Strictly speaking, two coordinate axes would suffice but the
        # math is cleaner with three axes.
        #
        # PyEphem measures the azimuth East (clockwise) from North.
        # The direction is standard but opposite to ISO spherical coordinates.
        try:
            xra, xdec = self._observer.radec_of(0, 0, fixed=False)
            yra, ydec = self._observer.radec_of(-np.pi / 2, 0, fixed=False)
            zra, zdec = self._observer.radec_of(0, np.pi / 2, fixed=False)
        except Exception as e:
            # Modified pyephem not available.
            # Translated pointing will include stellar aberration.
            xra, xdec = self._observer.radec_of(0, 0)
            yra, ydec = self._observer.radec_of(-np.pi / 2, 0)
            zra, zdec = self._observer.radec_of(0, np.pi / 2)
        self._observer.pressure = pressure
        xvec, yvec, zvec = ang2vec(
            np.pi / 2 - np.array([xdec, ydec, zdec]), np.array([xra, yra, zra])
        )
        # Orthonormalize for numerical stability
        xvec /= np.sqrt(np.dot(xvec, xvec))
        yvec -= np.dot(xvec, yvec) * xvec
        yvec /= np.sqrt(np.dot(yvec, yvec))
        zvec -= np.dot(xvec, zvec) * xvec + np.dot(yvec, zvec) * yvec
        zvec /= np.sqrt(np.dot(zvec, zvec))
        # Solve for the quaternions from the transformed axes.
        X = (xvec[1] + yvec[0]) / 4
        Y = (xvec[2] + zvec[0]) / 4
        Z = (yvec[2] + zvec[1]) / 4
        """
        if np.abs(X) < 1e-6 and np.abs(Y) < 1e-6:
            # Avoid dividing with small numbers
            c = .5 * np.sqrt(1 - xvec[0] + yvec[1] - zvec[2])
            d = np.sqrt(c**2 + .5 * (zvec[2] - yvec[1]))
            b = np.sqrt(.5 * (1 - zvec[2]) - c**2)
            a = np.sqrt(1 - b**2 - c**2 - d**2)
        else:
        """
        d = np.sqrt(np.abs(Y * Z / X))  # Choose positive root
        c = d * X / Y
        b = X / c
        a = (xvec[1] / 2 - b * c) / d
        # qarray has the scalar part as the last index
        quat = qa.norm(np.array([b, c, d, a]))
        """
        # DEBUG begin
        errors = np.array(
            [
                np.dot(qa.rotate(quat, [1, 0, 0]), xvec),
                np.dot(qa.rotate(quat, [0, 1, 0]), yvec),
                np.dot(qa.rotate(quat, [0, 0, 1]), zvec),
            ]
        )
        errors[errors > 1] = 1
        errors = np.degrees(np.arccos(errors))
        if np.any(errors > 1) or np.any(np.isnan(errors)):
            raise RuntimeError(
                "Quaternion is not right: ({}), ({} {} {})" "".format(errors, X, Y, Z)
            )
        # DEBUG end
        """
        return quat
    
    def to_JD(self, t):
        """
        Convert TOAST UTC time stamp to Julian date
        """
        return t / 86400.0 + 2440587.5

    def to_DJD(self, t):
        """
        Convert TOAST UTC time stamp to Dublin Julian date used
        by pyEphem.
        """
        return self.to_JD(t) - 2415020
    
    #these methods overwrite the original toast TOD class methods to get the data
    def _get(self, detector, start, n):
        name = "{}_{}".format("signal",detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref
    
    def _get_pntg(self, detector, start ,n):
        boresight = self._get_boresight(start, n)
        det_pntg = qa.mult(boresight, detquats()[detector])
        return det_pntg
        
    def _get_boresight(self, start, n):
        ref = self.cache.reference("boresight_quats")[start:start+n, :]
        return ref

    def _get_times(self, start, n):
        ref = self.cache.reference("timestamps")[start:start+n]
        return ref
    
    def _get_flags(self, detector, start, n):
        name = "{}_{}".format("flags", detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref
    
    def _get_common_flags(self, start, n):
        ref = self.cache.reference("common_flags")[start:start+n]
        return ref
    

        
  

    
    

    
