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
=====
Mask convention (0=good, 1=bad)
0 - keep
1 - mask out, i.e. output to ascii file
"""

from mantid.simpleapi import *
import numpy as np
import yaml

# INSTRUMENT NOMAD RELATED CONSTANTS
n_dets_per_tube = 128
mid_pos = 64
n_tubes = 792
n_tubes_per_pack = 8
n_packs = 99

# ---------------------- #
# - read the config file #
# ---------------------- #
with open("data/mask_gen_config.yml", 'r') as stream:
    config = yaml.safe_load(stream)
# figure out the state map per det
# - state 0 is the default state
# - state 1 is the full collimated state
# - state 2 is the half collimated state
states = np.zeros(n_packs, dtype=np.int)
full_col = np.array(config["collimation"]["full_col"], dtype=np.int)
half_col = np.array(config["collimation"]["half_col"], dtype=np.int)
states[full_col] = 1
states[half_col] = 2
states_per_tube = states.repeat(n_tubes_per_pack)
states_per_det = states_per_tube.repeat(n_dets_per_tube).reshape(
    (n_tubes, n_dets_per_tube))
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

# --------------------------------- #
# - get the per detector based mask #
# --------------------------------- #
# intensity_map2d[tube_id, det_pos]
# det_id = tube_id * n_dets_per_tube + det_pos
intensity_map2d = intensity_per_det.reshape((n_tubes, n_dets_per_tube))
median_per_tube = np.median(intensity_map2d, axis=1)
median1_per_tube = np.median(intensity_map2d[:, :mid_pos], axis=1)
median2_per_tube = np.median(intensity_map2d[:, mid_pos:], axis=1)
mask_per_det = np.ones(intensity_map2d.shape, dtype=np.bool)
# check [2], standard tubes (state id is 0)
check2 = (intensity_map2d - median_per_tube[:, np.newaxis] * low_pixel) * (
    intensity_map2d - median_per_tube[:, np.newaxis] * high_pixel) > 0
mask_per_det = np.where(states_per_det == 0, check2, mask_per_det)
# check [3], full collimated tubes (state id is 1)
low_bound = 1 + (low_pixel - 1) * 3
high_bound = 1 + (high_pixel - 1) * 3
check3 = (intensity_map2d - median_per_tube[:, np.newaxis] * low_bound) * (
    intensity_map2d - median_per_tube[:, np.newaxis] * high_bound) > 0
mask_per_det = np.where(states_per_det == 1, check3, mask_per_det)
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
mask_per_det = np.where(states_per_det == 2, check4, mask_per_det)
# flatten and ready
mask_per_det = mask_per_det.flatten()

# - get the per tube based mask
# NOTE: due to the irregular size of each bank, we have to loop
intensity_per_tube = np.sum(intensity_map2d, axis=1)
mask_per_tube = np.ones(intensity_per_tube.shape, dtype=np.bool)
# calculate the median per tube
median_per_tube = np.zeros_like(intensity_per_tube, dtype=np.int)
# || use bank level median
# -> for all 8-packs in flat panels, i.e. bank5 and bank6
# -> for non-collimated 8-packs in non-flat panels, i.e. bank 1 - 4
states_per_det = states_per_det.flatten()
# NOTE: there are only six banks in NOMAD
for i in range(6):
    _cfg = config["bank"][f"bank_{i+1}"]
    _st_tube = _cfg[0] * n_tubes_per_pack
    _st_det = _st_tube * n_dets_per_tube
    _ed_tube = (_cfg[1] + 1) * n_tubes_per_pack
    _ed_det = _ed_tube * n_dets_per_tube
    int_tmp = intensity_per_det[_st_det:_ed_det]
    if i > 0:
        # for bank2-6, we exclude the collimated pack from median calculation
        # i.e. keeping det with state 0 only
        states_tmp = states_per_det[_st_det:_ed_det]
        int_tmp = int_tmp[states_tmp == 0]
    # calculate the median
    median_per_tube[_st_tube:_ed_tube] = np.median(
        intensity_per_det[_st_det:_ed_det])
# || use 8-pack level median
# -> for collimated 8-packs in non-flat panels, i.e. bank 1 - 4, the median
#    need to be calculated on a per 8-packs basis
#    so basically we just need to go through all the fully collimated 8-packs,
#    which all happens to be within bank 1 - 4, and update the median
for i in full_col:
    _st_tube = i * n_tubes_per_pack
    _st_det = _st_tube * n_dets_per_tube
    _ed_tube = (i + 1) * n_tubes_per_pack
    _ed_det = _ed_tube * n_dets_per_tube
    median_per_tube[_st_tube:_ed_tube] = np.median(
        intensity_per_det[_st_det:_ed_det])
# now performing check
# || check flat panels, i.e. bank 5 and bank 6
# NOTE: For tubes that are not listed in the bank, their median is set to 0,
#       which makes sure that check9 will have False for them.
#       Therefore, we don't have to worry about kick out tubes that are not
#       listed in the bank.
check9 = intensity_per_tube < 0.1 * median_per_tube
st_tube = config["bank"]["bank_5"][0] * n_tubes_per_pack
ed_tube = (config["bank"]["bank_6"][1] + 1) * n_tubes_per_pack
mask_per_tube[st_tube:ed_tube] = check9[st_tube:ed_tube]
# || check non-flat panels, i.e. bank 1 - 4
check78 = (intensity_per_tube - low_tube * median_per_tube) * (
    intensity_per_tube - high_tube * median_per_tube) > 0
# NOTE: now we need to make sure we don't kick out all the zero median tubes since
#       there are not part of the check
check78 = np.where(median_per_tube > 0, check78, False)
st_tube = config["bank"]["bank_1"][0] * n_tubes_per_pack
ed_tube = (config["bank"]["bank_4"][1] + 1) * n_tubes_per_pack
mask_per_tube[st_tube:ed_tube] = check78[st_tube:ed_tube]
# extend to detector
mask_per_tube = mask_per_tube.repeat(n_dets_per_tube)

# final results
mask_final = np.logical_or(mask_per_det, mask_per_tube)

# write to file
det_to_mask = np.argwhere(mask_final == True)
np.savetxt("mask.out", det_to_mask, fmt="%u", delimiter="\n")
