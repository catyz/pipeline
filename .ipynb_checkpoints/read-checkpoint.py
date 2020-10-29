#!/usr/bin/env python3

import os
import argparse
from spt3g import core, dfmux, gcp
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime as dt
import matplotlib.dates as md


def dump(run_number, dir_output):
    ddir = '/global/cscratch1/sd/yzh/data/ChileData/Run2000{:04d}/'.format(run_number)
    dfiles = sorted(glob(ddir + '*.g3'))

    plotdir = '{:s}/Run{:04d}/'.format(dir_output, run_number)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    # peek and get the number of detectors
    f = core.G3File(dfiles[0])
    frame = f.next()
    while frame.type != core.G3FrameType.Scan:
        frame = f.next()
    n_dets = len(frame['RawTimestreams_I'].keys())

    # loading lots of tods is ram intensive, so do it in chunks
    jump_interval = 100
    for start in range(n_dets)[::jump_interval]:
        end = start + jump_interval

        # detector TOD, in both phases
        tod_buffs_I = {}
        tod_buffs_Q = {}
        def get_TOD(frame, start, end):
            if frame.type == core.G3FrameType.Scan:
                for det in frame['RawTimestreams_I'].keys()[start:end][::-1]: # TEMP
                    print('Processing {:s}...'.format(det))
                    if det not in tod_buffs_I.keys():
                        tod_buffs_I[det] = []
                        tod_buffs_Q[det] = []
                    tod_buffs_I[det].append(frame['RawTimestreams_I'][det])
                    tod_buffs_Q[det].append(frame['RawTimestreams_Q'][det])

        # detector sample times
        times_buff = []
        def get_times(frame):
            if frame.type == core.G3FrameType.Scan:
                times_buff.append(np.array([t.time * 1.e-8 for t in frame['DetectorSampleTimes']]))

        # boresight encoder data
        antennatime_buff = []
        az_buff = []
        el_buff = []
        def get_azel(frame):
            if frame.type == core.G3FrameType.Scan:
                antennatime_buff.append([frame['TrackerStatus'].time[i].time * 1.e-8  for i in range(len(frame['TrackerStatus'].time))])
                az_buff.append(frame['TrackerStatus'].az_pos)
                el_buff.append(frame['TrackerStatus'].el_pos)

        pipe = core.G3Pipeline()
        pipe.Add(core.G3Reader, filename=dfiles)
        pipe.Add(get_times)
        pipe.Add(get_TOD, start=start, end=end)
        pipe.Add(get_azel)
        pipe.Run()

        # group stuff into single arrays
        times = np.concatenate(times_buff)
        antennatimes = np.concatenate(antennatime_buff)
        az = np.concatenate(az_buff)
        el = np.concatenate(el_buff)

        # nice date format for plotting
        bolodates = md.date2num([dt.datetime.fromtimestamp(t) for t in times])
        antennadates = md.date2num([dt.datetime.fromtimestamp(t) for t in antennatimes])

        for det in tod_buffs_I:
            print('Concatenating {:s}...'.format(det))
            tod_I = np.concatenate(tod_buffs_I[det])
            tod_Q = np.concatenate(tod_buffs_Q[det])

            # skip problem TODs
            if np.all(np.isnan(tod_I)) or np.all(np.isnan(tod_Q)): continue
            if len(tod_I) != len(times) or len(tod_I) != len(tod_Q):
                print('skipping because tod is different length')
                continue

            fig, ax = plt.subplots(figsize=(12,10), sharex=True, nrows=4, ncols=1)
            for iax in range(4):
                ax[iax].xaxis.set_major_formatter(md.DateFormatter('%Y/%m/%d %H:%M:%S'))
                plt.xticks(rotation=45, fontsize=6)
                #ax[iax].xaxis.set_tick_params(which='both', rotation=45)
            ax[0].plot(bolodates, tod_I, 'b.', markersize=2)
            ax[1].plot(bolodates, tod_Q, 'r.', markersize=2)
            ax[2].plot(antennadates, az*180/np.pi, 'c-')
            ax[3].plot(antennadates, el*180/np.pi, 'm-')
            ax[3].set_xlabel('UTC time')
            ax[0].set_ylabel('ADC counts (I)')
            ax[1].set_ylabel('ADC counts (Q)')
            ax[2].set_ylabel('Boresight Az [deg]')
            ax[3].set_ylabel('Boresight El [deg]')
            fig.suptitle('Run {:d} - {:s}'.format(run_number, det), fontweight='bold')
            fig.savefig(plotdir + '{:s}.png'.format(det))
            #plt.show()
            plt.close(fig)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_number', action='store', type=int, help='run number')
    parser.add_argument('dir_output', action='store', help='path to directory for dump')
    args = parser.parse_args()
    dump(args.run_number, args.dir_output)

if __name__ == '__main__':
    main()