import os
import re
import sys
import numpy as np
import sqlite3
from toast_pb2 import pb2tod

#creates the tod object. I've skipped all the MPI stuff for now
def create_tods(datafolder):
    tods = []
       
    tods.append(
        pb2tod(
            datafolder
        )
    )
    return tods

#makes the observation dictionary with some made up metadata
def make_observations(data, tods):
    weather = None

    for iobs, tod in enumerate(tods):
        ob = {}
        ob["name"] = tod.name
        ob["id"] = tod.ces_id
        ob["telescope_id"] = 12345
        ob["site_id"] = 678910
        #meta = tod.meta()
        #ob["altitude"] = meta["site_alt"]
        ob["weather"] = weather
        ob["fpradius"] = 2.26
        ob["tod"] = tod
        #if args.turnarounds:
            # With the filter, we want continuous baselines even
            # across the turnarounds
            #ob["intervals"] = [tod.valid_interval]
        #else:
            #ob["intervals"] = tod.valid_intervals
        ob["baselines"] = None
        ob["noise"] = tod.noise
        data.obs.append(ob)
    return


