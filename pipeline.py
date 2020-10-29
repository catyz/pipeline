import numpy as np
import argparse

from simons_array_python import sa_pipeline_filters as sa_pf
from simons_array_python import sa_pipeline_inputs as sa_pi
from simons_array_python import sa_tod as sa_tod
from simons_array_python import sa_observation as sa_ob
from simons_array_python import sa_config
from simons_array_python import sa_sql
from simons_array_python import sa_pointing as sa_p
from simons_array_python import sa_timestream_operators as sa_op

from simons_array_python import sa_toast_pipeline_tools as sa_tpt

import toast.pipeline_tools as tpt

parser = argparse.ArgumentParser(
        description="SA + TOAST simulations example pipeline. Run with python3 pipeline.py @pars",
        fromfile_prefix_chars="@",
    )

#Add arguments for toast operators inputs
sa_tpt.add_general_args(parser)
tpt.add_noise_args(parser)
tpt.add_atmosphere_args(parser)
tpt.add_sss_args(parser)
tpt.add_sky_map_args(parser)
tpt.add_pointing_args(parser)
tpt.add_mapmaker_args(parser)

args = parser.parse_args()

#Set the name of the data tod 
data_prefix = 'data'

############################### NORMAL SA PIPELINE  ##############################################
run_id = args.run_id
subrun_id = args.subrun_id

obs = sa_ob.Observation((run_id,subrun_id)) 
obs.detectors = ['13.10_112.90B'] 
obs.load_metadata()

# len_sec = (obs.last_mjd-obs.first_mjd)*24*3600
# obs.len_sec = len_sec
# obs.scan_freq = 5

pi = sa_pi.InputLevel0CachedByObsID(
    all_detectors = obs.detectors,
    n_per_cache = 1,
    load_slowdaq = False,
    load_hwp = False,
    load_dets = True, 
    load_g3 = True,
    ignore_faulty_frame = True,
    record_frame_time = True,
    input_name = data_prefix, #For TOAST purposes, default is 'data' anyway
)

sa_operator_stack = sa_pf.OperatorComposite(
    sa_p.ComputeBoresightQuaternions(),
    sa_pf.OperatorTelescopeDataInterpolator(prefix='corrected_'),
    sa_pf.OperatorScanCorrector('raw_scan_flag', 'raw_el_pos', 'raw_antenna_time_mjd'),
    sa_pf.OperatorScanCorrector('raw_scan_flag', 'raw_az_pos', 'raw_antenna_time_mjd'),
    sa_pf.OperatorDataInitializer(pi),
)
sa_operator_stack.filter_obs(obs) 
############################### NORMAL SA PIPELINE  ##############################################


#Here begins the simulation part (TOAST)

#Set names of simulated tods
sims_prefix = 'mc'    

#Pack the toast data object with everything the operators need
data, comm = sa_tpt.make_toast_data(args, obs_list=[obs], sims_prefix=sims_prefix, data_prefix=data_prefix)

mc_start = args.mc_start #Starting random seed for toast simulations
nsims = args.nsims #Number of sims to run
outpath = args.outpath #Out directory for maps if using toast mapmaker

#Below is just an example of what you could run
tpt.expand_pointing(args, comm, data)

for mc in range(mc_start, mc_start + nsims):
    
    total_prefix = sims_prefix+str(mc)        

    print(f'Processing {total_prefix}')
    
#    tpt.simulate_atmosphere(args, comm, data, mc, total_prefix) #Make sure to delete './atm_cache' when using new  random seeds/observations
    
#    tpt.scale_atmosphere_by_frequency(args, comm, data, freq=None, mc=mc, cache_name=total_prefix) 

    tpt.scan_sky_signal(args, comm, data, total_prefix, mc=mc)

#    tpt.simulate_noise(args, comm, data, mc, total_prefix)

#    tpt.simulate_sss(args, comm, data, mc, total_prefix)  

#     tpt.apply_mapmaker(args, comm, data, outpath, total_prefix) #Actually filters the tods, so don't run this if you just want the tods

#Appends -I to the cachenames of simulated tods to match SA format
sa_tpt.add_suffix_to_detname(data, sims_prefix, data_prefix, suffix='-I')

#Makeshift SA operator, just loops through tods and plots them
# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(2,1, sharex=True)

# for obs in data.obs:
#     for tod in obs['tod']:
#         signal = tod.read('13.10_112.90B-I')
#         if tod._source_prefix == data_prefix+'_':
#             ax[0].plot(signal)
#         else:
#             ax[1].plot(signal)

# fig.savefig('tods.png')