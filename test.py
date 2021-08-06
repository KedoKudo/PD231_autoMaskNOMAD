#!/usr/bin/env python
"""
For a fake instrument with 20 pixels that consists of 5 tubes, 4 pixels each
a = np.arange(20)
a = a.reshape(5,4)
>> 
array([[ 0,  1,  2,  3],  -> tube 1
       [ 4,  5,  6,  7],  -> tube 2
       [ 8,  9, 10, 11],  -> tube 3
       [12, 13, 14, 15],  -> tube 4
       [16, 17, 18, 19]]) -> tube 5
c = dummy(a) # operation that retain the shape of a
c.flatten()
>>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19])
"""

from mantid.simpleapi import *
import numpy as np
import yaml

# INSTRUMENT NOMAD RELATED CONSTANTS
n_dets_per_tube = 128
mid_pos = 64
n_tubes = 792
n_tubes_per_bank = 8
n_banks = 99

# ---------------------- #
# - read the config file #
# ---------------------- #
with open("data/mask_gen_config.yml", 'r') as stream:
    config = yaml.safe_load(stream)
# figure out the state map per det
# - state 0 is the default state
# - state 1 is the collimated state
# - state 2 is the non-collimated state
states = np.zeros(n_banks, dtype=np.int)
full_col = np.array(config["collimation"]["full_col"], dtype=np.int)
half_col = np.array(config["collimation"]["half_col"], dtype=np.int)
states[full_col] = 1
states[half_col] = 2
states = states.repeat(n_tubes_per_bank)
# get the desired threshold
low_pixel = float(config["threshold"]["low_pixel"])
high_pixel = float(config["threshold"]["high_pixel"])
low_tube = float(config["threshold"]["low_tube"])
high_tube = float(config["threshold"]["high_tube"])

# ------------------ #
# - read in the data #
# ------------------ #
ws = Load(
    Filename='data/NOM_147298.nxs.h5',
    OutputWorkspace='NOM_147298.nxs',
    NumberOfBins=1,  # sum spectrum for each det/pix
)
# extract sum_per_pix
intensity_per_det = ws.extractY()
# calculate the solid angle
ws_solidAngle = SolidAngle(InputWorkspace=ws)
solidAngle = ws_solidAngle.extractY()
# normalize the intensity with solid angle
intensity_per_det /= solidAngle

# intensity_map2d[tube_id, det_pos]
# det_id = tube_id * n_dets_per_tube + det_pos
intensity_map2d = intensity_per_det.reshape((n_tubes, n_dets_per_tube))
median_per_tube = np.median(intensity_map2d, axis=1)
median1_per_tube = np.median(intensity_map2d[:, :mid_pos], axis=1)
median2_per_tube = np.median(intensity_map2d[:, mid_pos:], axis=1)

# --------------------------------- #
# - get the per detector based mask #
# --------------------------------- #
mask_per_det = np.ones(intensity_map2d.shape, dtype=np.bool)
# check [2], standard tubes (state id is 0)
check2 = (intensity_map2d - median_per_tube[:, np.newaxis] * low_pixel) * (
    intensity_map2d - median_per_tube[:, np.newaxis] * high_pixel) > 0
mask_per_det = np.where(states == 0, check2, mask_per_det)
# check [3], full collimated tubes (state id is 1)
low_bound = 1 + (low_pixel - 1) * 3
high_bound = 1 + (high_pixel - 1) * 3
check3 = (intensity_map2d - median_per_tube[:, np.newaxis] * low_bound) * (
    intensity_map2d - median_per_tube[:, np.newaxis] * high_bound) > 0
mask_per_det = np.where(states == 1, check3, mask_per_det)
# check [4], half collimated tubes (state id is 2)
# NOTE: reuse the same bound here
# low_bound = 1 + (low_pixel-1)*3
# high_bound = 1 + (high_pixel-1)*3
check4_top = (intensity_map2d - median1_per_tube[:, np.newaxis] *
              low_bound) * (intensity_map2d -
                            median1_per_tube[:, np.newaxis] * high_bound) > 0
check4_bot = (intensity_map2d - median2_per_tube[:, np.newaxis] *
              low_bound) * (intensity_map2d -
                            median2_per_tube[:, np.newaxis] * high_bound) > 0
check4 = check4_top
check4[:, mid_pos:] = check4_bot[:, mid_pos:]
mask_per_det = np.where(states == 2, check4, mask_per_det)
# flatten and ready
mask_per_det = mask_per_det.flatten()

# - get the per tube based mask
intensity_per_tube = np.sum(intensity_per_det, axis=0)
mask_per_tube = np.ones(intensity_per_det.shape, dtype=np.bool)