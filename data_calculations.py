import glob, os, sys, timeit
import matplotlib
import numpy as np
import pandas as pd

sys.path.append('../')
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.constants import c


LAM_LAB_CIV = 1549.06       # Å, adopted rest wavelength of C IV
C_KMS = c.to('km/s').value
LN10 = np.log(10.0)

# Zuo+2020 calibration constants
ALPHA_BS = 0.67             # Eq. 6 (blueshift relation)
BETA_BS = 0.41
ALPHA_AS = 2.03             # Eq. 7 (asymmetry relation)
BETA_AS = -1.1

# VP06 mass calibration
A = 0.66
B = 0.53
C = 2.0

base_path = "/Users/nayera/PyQSOFit/"

def get_L1350_from_fits(fits_path):
    """
    Read continuum info from a QSOFit output FITS file and return
    L1350 and its uncertainty.

    Computes log10(L1350/erg s^-1) from L1450 and PL_slope in the FITS file:

        logL1350 = logL1450 + (alpha + 1) * log10(1350/1450)

    where alpha = PL_slope (continuum slope in F_lambda ~ lambda^alpha)

    Returns both logL1350 and propagated error.
    """
    with fits.open(fits_path) as f:
        tab = f[1].data  
        row = tab[0]

        # Get L1450 and power-law slope alpha
        logL1450 = float(row['L1450'])
        logL1450_err = float(row['L1450_err'])
        alpha = float(row['PL_slope'])
        alpha_err = float(row['PL_slope_err'])

        # calculate logL1350
        ratio = 1350.0 / 1450.0
        log_ratio = np.log10(ratio)
        logL1350 = logL1450 + (alpha + 1.0) * log_ratio

        # Error propagation in log-space
        logL1350_err = np.sqrt(logL1450_err**2 + (log_ratio * alpha_err)**2)

    return logL1350, logL1350_err
    

def blueshift_calc(lam_half, lam_lab=LAM_LAB_CIV):
    """
    Compute C IV blueshift (km/s) using λ_half (rest frame).
    ΔV = c * (λ_lab - λ_half) / λ_lab
    """
    return C_KMS * (lam_lab - lam_half) / lam_lab


def asym_calc(lam_blue, lam_red, lam_peak):
    """
    Compute Zuo et al. asymmetry parameter A_S:

        A_S = ln[ (λ_red - λ_peak) / (λ_peak - λ_blue) ]

    Returns np.nan if the geometry is invalid.
    """
    num = lam_red - lam_peak
    den = lam_peak - lam_blue
    if num <= 0 or den <= 0:
        return np.nan
    return np.log(num / den)

def bhm_civ_calc(logL1350, FWHM):
    """
    Compute log10(M_BH/M_sun) from C IV using
    Vestergaard & Peterson (2006) calibration:

        log(M/Msun) = a + b * log10(L1350/1e44) + c * log10(FWHM/km s^-1)

    where a = 0.66, b = 0.53, c = 2.0
    """
    return A + B * (logL1350 - 44.0) + C * np.log10(FWHM)

def bhm_civ_error(logL1350_err, FWHM, FWHM_err):
    """
    Propagate uncertainties for log10(M_BH/M_sun) assuming
    uncorrelated errors in L1350 and FWHM.

    Using:

        σ_M^2 = (0.53*σ_logL)^2 + (2/(ln(10)*FWHM)*σ_FWHM)^2
    """
    term_L = B * logL1350_err
    term_F = (C / (LN10 * FWHM)) * FWHM_err

    return np.sqrt(term_L**2 + term_F**2)

# Correction functions for blueshift and asymmetry

def fwhm_blueshift_corrected(FWHM, blueshift):
    # From Zuo2020 Eq. 6 (only corrects if positive blueshift and valid denominator)
    if blueshift <= 0 or np.isnan(blueshift):
        return FWHM
    denom = ALPHA_BS + BETA_BS * (blueshift/1000.0)
    if denom <=0:
        return FWHM
    return FWHM / denom

def fwhm_asym_corrected(FWHM, asym):
    # From Zuo2020 Eq. 7 (only corrects if valid denominator)
    if np.isnan(asym):
        return FWHM
    denom = ALPHA_AS + BETA_AS*asym
    if denom <=0:
        return FWHM
    return FWHM/denom

def bhm_civ_corrected(logL1350, FWHM_corr):
    # Same calculation as above just with corrected FWHM
    return A + B * (logL1350 - 44.0) + C * np.log10(FWHM_corr)

def main(input_csv, output_csv):
    """
    Read input CSV with pandas, compute derived quantities,
    and write a new CSV with extra columns.
    """
    df = pd.read_csv(input_csv)

    params = [ 
        'logL1350', 'logL1350_err', 'CIV_blueshift_kms', 
        'CIV_asym', 'logMBH_CIV', 'logMBH_CIV_err', 
        'FWHM_CIV_corr_blueshift','logMBH_CIV_corr_blueshift',
        'FWHM_CIV_corr_asym','logMBH_CIV_corr_asym'
    ]

    # Initial empty params
    for param in params:
        df[param] = np.nan

    for idx, row in df.iterrows():
        fits_file = row['fits_path']
        fits_path = base_path + "CIV_galaxy_spectra_targets/" + row['Target'] + "/output/" + fits_file + ".fits"

        # Get L1350
        logL1350, logL1350_err = get_L1350_from_fits(fits_path)

        # Basic quantities from CSV
        lam_blue = row['lam_blue']
        lam_red = row['lam_red']
        lam_half = row['lam_half']
        lam_peak = row['br_peak']          
        FWHM = row['FWHM_CIV']
        FWHM_err = row['FWHM_err_mcmc']

        # Derived quantities
        blueshift = blueshift_calc(lam_half)
        asym      = asym_calc(lam_blue, lam_red, lam_peak)
        logM      = bhm_civ_calc(logL1350, FWHM)
        logM_err  = bhm_civ_error(logL1350_err, FWHM, FWHM_err)

        # Corrected quantities
        FWHM_corr_bs = fwhm_blueshift_corrected(FWHM, blueshift)
        logM_corr_bs = bhm_civ_corrected(logL1350, FWHM_corr_bs)

        FWHM_corr_as = fwhm_asym_corrected(FWHM, asym)
        logM_corr_as = bhm_civ_corrected(logL1350, FWHM_corr_as)

        # Store back into dataframe
        df.loc[idx, 'logL1350'] = logL1350
        df.loc[idx, 'logL1350_err'] = logL1350_err
        df.loc[idx, 'CIV_blueshift_kms'] = blueshift
        df.loc[idx, 'CIV_asym'] = asym
        df.loc[idx, 'logMBH_CIV'] = logM
        df.loc[idx, 'logMBH_CIV_err'] = logM_err
        df.loc[idx, "FWHM_CIV_corr_blueshift"] = FWHM_corr_bs
        df.loc[idx, "logMBH_CIV_corr_blueshift"] = logM_corr_bs
        df.loc[idx, "FWHM_CIV_corr_asym"] = FWHM_corr_as
        df.loc[idx, "logMBH_CIV_corr_asym"] = logM_corr_as

    # Write output
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    input_csv  = base_path + "Target_lists/target_list_info_for_calc.csv"
    output_csv = base_path + "Target_lists/civ_output_with_derived.csv"
    main(input_csv, output_csv)