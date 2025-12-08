import glob, os, sys, timeit
import matplotlib
import numpy as np

sys.path.append('../')
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.constants import c

'''
To read an astronomical .fits spectrum, use Python with astropy.io.fits, 
open the file (fits.open()), inspect with .info(), access data (like flux and 
wavelength arrays from specific HDUs, e.g., f[0].data), and plot using matplotlib
'''

lam_lab = 1549.06
lam_blue = 
lam_red = 
lam_half = 1528.92000
lam_peak = 
sun_mass = 

def lum_calc():
    f = fits.open('/Users/nayera/PyQSOFit/CIV_galaxy_spectra_targets/J1050+4627/output/B_23spec-6664-56383-0572.fits')
    #data = fits.info('/Users/nayera/PyQSOFit/CIV_galaxy_spectra_targets/J0759+1800/output/B_20spec-4490-55629-0074.fits')

    # cols = f[1].columns
    # print(cols.names)

    L1350 = f[1].data['L1450']   # example, column name may differ
    return L1350

def blue_calc():
    blueshift = c * (lam_lab - lam_half)/lam_lab
    return blueshift


def asym_calc():
    asym = np.log((lam_red - lam_peak)/(lam_peak - lam_blue))
    return asym

def bhm_calc():
    term1 = 0.66 # a value from Vestergaard & Peterson (2006) calibration
    term2 = 0.53 * np.log10(lum_calc()/(10**44))
    term3 = 2.0 * np.log10(FWHM)
    bhm = term1 + term2 + term3
    return np.log10(bhm/sun_mass)