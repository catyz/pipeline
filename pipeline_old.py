from toast_pb2 import pb2tod
import toast
from pipeline_tools import * 
import toast.todmap
import toast.pipeline_tools
import argparse
from toast.todmap import (
    OpPointingHpix,
    OpAccumDiag,
    OpSimScan,
    OpSimPySM
)
from toast.map import (
    DistPixels
)
import pickle
from toast.tod import OpSimNoise

print(toast.__file__)
#detector = 'PB20.13.10_Comb30Ch28'
#boloname = '13.10_248.150T'

parser = argparse.ArgumentParser(
    description='Make a healpix map of a given observation'
)

parser.add_argument(
    "--nside", required=False, type=np.int, default=512, help="Map resolution",
)

toast.pipeline_tools.add_polyfilter_args(parser)

args = parser.parse_args()

#data location is hard coded for now
#datafolder = '/global/cscratch1/sd/yzh/data/ChileData/g3compressed/20100000/Run20100857/' 
#datafolder = '/global/cscratch1/sd/yzh/data/Run20100665/' #jupiter
datafolder = '/global/cscratch1/sd/yzh/data/ChileData/Run20000065/' #venus

#load the focalplane
#fp = pickle.load(open('focalplane.pkl','rb'))

#create the toast data object
comm = toast.Comm(groupsize=1)
data = toast.Data(comm)

tods = create_tods(datafolder)
make_observations(data, tods)


print('making pointing matrix')
pointing = toast.todmap.OpPointingHpix(
    nside=args.nside,
    nest=True,
    mode="I",
    pixels="pixels",
    weights="weights"
)

pointing.exec(data)
# if args.apply_polyfilter:
#     toast.pipeline_tools.apply_polyfilter(args, comm, data, "signal")


# print('simulating pysm')    
# pysm_sky_config = ["c1"]
    
# opsim_pysm = OpSimPySM(
#     data,
#     comm=None,
#     out='scan',
#     pysm_model=pysm_sky_config,
#     apply_beam=True,
#     debug=True,
#     focalplanes=[fp],
#     nest=True,
#     pixels='pixels'
# )
# opsim_pysm.exec(data)

print('simulating detector noise')
op_simnoise = OpSimNoise(
    out = 'noise',
    realization = 0,
    component = 0,
    noise = 'noise'
)
op_simnoise.exec(data)



print('scan input map (instead of pysm)')
distmap = DistPixels(data, nnz=1, dtype=np.float32, pixels='pixels')
distmap.read_healpix_fits('input_map.fits')

op_simscan = OpSimScan(
    distmap = distmap,
    pixels = 'pixels',
    weights = 'weights',
    out = 'scan'
)
op_simscan.exec(data)

print('writing stuff to file')
tod = data.obs[0]['tod']
tod.cache.reference('noise_13.12_67.90T').tofile('noise_tod.npy')
tod.cache.reference('scan_13.12_67.90T').tofile('synth_tod.npy')
tod.cache.reference('signal_13.12_67.90T').tofile('actual_tod.npy')

# synth_tod = 
# actual_tod = tod.cache.reference('signal_13.12_67.90T')
#noise_tod = tod.cache.reference('noise_13.12_67.90T')

# pickle.dump(synth_tod, open("synth_tod.pkl","wb"))
# pickle.dump(actual_tod, open("actual_tod.pkl","wb"))
#pickle.dump(noise_tod, open("noise_tod.pkl","wb"))

# import toast.qarray as  qa
pix = tod.cache.reference('pixels_13.12_67.90T')
weights = tod.cache.reference('weights_13.12_67.90T')


# theta, phi, pa = qa.to_angles(tod.read_pntg(detector='13.12_67.90T'))
pickle.dump(pix, open('pix.pkl','wb'))
pickle.dump(weights, open('weights.pkl','wb'))

# pickle.dump([theta,phi,pa], open('angles.pkl', 'wb'))

# sky = tod.cache.reference('sky_13.12_67.90T')


print('making map')
mapmaker = toast.todmap.OpMapMaker(
    nside=args.nside,
    nnz=1,
    name=None,
    pixels="pixels",
    intervals=None,
    baseline_length=None,
    use_noise_prior=False,
    outdir="./map",
)

# #madam instead of normal one
# mapmaker = toast.todmap.OpMadam(
#         params={
#             "write_matrix" : False,
#             "write_wcov" : False,
#             "write_hits" : True,
#             "write_binmap" : True,
#             "write_map" : False,
#             "kfirst" : False,
#             "nside_map" : args.nside,
#             "file_root" : "madam",
#             "path_output" : "./map",
#         },
#         name="scan",
#     )

mapmaker.exec(data)



print('DONE YAY WOOHOO')
