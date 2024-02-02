# module input
'''
User-defined control inputs for quantum numbers and plotting, among others.
'''

import pandas as pd
import numpy as np

# global temperature [K]
TEMP: float = 300.0
# global pressure    [Pa]
PRES: float = 101325.0

# molecule to simulate
MOLECULE: str = 'o2+'

# upper and lower electronic states
STATE_UP: str = 'a2pu'
STATE_LO: str = 'x2pg'

# NOTE: temperature used in Cosby (0, 9) is 300 K, pressure is 2666.45 Pa

# predissociation constants
PD_DATA = pd.read_csv('../data/predissociation.csv', delimiter=' ')

# Franck-Condon factors
FC_DATA = np.loadtxt('../data/franck-condon/cheung_rkr_fc.csv', delimiter=' ')

# rotational levels
# Cosby predissociation data only goes up to N = 36
ROT_LVLS: np.ndarray = np.arange(0, 25, 1)

# list of vibrational transitions considered in (v', v'') format
VIB_BANDS: list[tuple[int, int]] = [(4, 20)]

# band origin override
BAND_ORIG: tuple[bool, int] = (False, 36185)

# line data
LINE_DATA: bool = True

# printing line info
PRINT_INFO: bool  = True
INFO_LIMS:  tuple = (11950, 12250)

# convolve data separately (convolve each vibrational transition individually)
CONV_SEP:  bool = False

# instrument function
INST_SEP: bool = False
INST_ALL: bool = False

# convolve data together (combine all quantized line positions and convolve together)
CONV_ALL:  bool = False

# granulatity of the convolved data
CONV_GRAN: int  = 50000

# sample data
SAMP_DATA: bool = True
SAMP_FILE: list[str] = ['o2+/chen']
SAMP_COLS: list[str] = ['purple']
SAMP_LABL: list[str] = ['pgopher']

# general plotting
PLOT_SAVE:  bool  = False
PLOT_PATH:  str   = '../img/example.webp'
DPI:        int   = 96
SCREEN_RES: tuple = (1920, 1080)

# custom plot limits
SET_LIMS: tuple[bool, tuple] = (False, (36170, 36192))
# custom font size
FONT_SIZE: tuple[bool, int] = (False, 18)
