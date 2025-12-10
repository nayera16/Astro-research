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
base_path = "nayera/PyQSOFit/"

def get_L1350_from_fits(fits_path):
    """
    Read continuum info from a QSOFit output FITS file and return
    L1350 and its uncertainty.

    Compute L1350 from L1450 and PL_slope:
        L_lambda ∝ λ^alpha
        L1350 = L1450 * (1350/1450)^alpha

    Error propagation:
        σ(L1350)/L1350 = sqrt[ (σ_L1450/L1450)^2 + (ln(1350/1450)*σ_alpha)^2 ]
    """
    with fits.open(fits_path) as f:
        tab = f[1].data  
        row = tab[0]

        # Get L1450 and power-law slope alpha
        L1450      = float(row['L1450'])
        L1450_err  = float(row['L1450_err'])
        alpha      = float(row['PL_slope'])
        alpha_err  = float(row['PL_slope_err'])

        lam1 = 1450.0
        lam2 = 1350.0

        # L1350 from power law
        ratio = lam2 / lam1
        L1350 = L1450 * ratio**alpha

        # Error propagation in log-space
        ln_ratio = np.log(ratio)
        rel_err_sq = (L1450_err / L1450)**2 + (ln_ratio * alpha_err)**2
        L1350_err = L1350 * np.sqrt(rel_err_sq)

    return L1350, L1350_err
    

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

def bhm_civ_calc(L1350, FWHM):
    """
    Compute log10(M_BH/M_sun) from C IV using
    Vestergaard & Peterson (2006) calibration:

        log(M/Msun) = a + b * log10(L1350/1e44) + c * log10(FWHM/km s^-1)

    where a = 0.66, b = 0.53, c = 2.0
    """
    a = 0.66
    b = 0.53
    c_exp = 2.0

    term1 = a
    term2 = b * np.log10(L1350 / 1e44)
    term3 = c_exp * np.log10(FWHM)

    return term1 + term2 + term3

def bhm_civ_error(L1350, L1350_err, FWHM, FWHM_err):
    """
    Propagate uncertainties for log10(M_BH/M_sun) assuming
    uncorrelated errors in L1350 and FWHM.

    For y = a + b*log10(L) + c*log10(F),
        σ_y^2 = (b / (ln(10)*L) * σ_L)^2 + (c / (ln(10)*F) * σ_F)^2
    """
    a = 0.66
    b = 0.53
    c_exp = 2.0

    # Guard against zeros or negative values
    if L1350 <= 0 or FWHM <= 0:
        return np.nan

    term_L = b * L1350_err / (L1350 * LN10)
    term_F = c_exp * FWHM_err / (FWHM * LN10)

    return np.sqrt(term_L**2 + term_F**2)

def main(input_csv, output_csv):
    """
    Read input CSV with pandas, compute derived quantities,
    and write a new CSV with extra columns.
    """
    df = pd.read_csv(input_csv)

    # Initial empty params
    df['L1350'] = np.nan
    df['L1350_err'] = np.nan
    df['CIV_blueshift_kms'] = np.nan
    df['CIV_asym'] = np.nan
    df['logMBH_CIV'] = np.nan
    df['logMBH_CIV_err'] = np.nan

    for idx, row in df.iterrows():
        fits_file = row['fits_path']
        fits_path = base_path + "CIV_galaxy_spectra_targets/" + row['Target'] + "/output/" + fits_file

        # Get L1350
        L1350, L1350_err = get_L1350_from_fits(fits_path)

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
        logM      = bhm_civ_calc(L1350, FWHM)
        logM_err  = bhm_civ_error(L1350, L1350_err, FWHM, FWHM_err)

        # Store back into dataframe
        df.loc[idx, 'L1350']             = L1350
        df.loc[idx, 'L1350_err']         = L1350_err
        df.loc[idx, 'CIV_blueshift_kms'] = blueshift
        df.loc[idx, 'CIV_asym']          = asym
        df.loc[idx, 'logMBH_CIV']        = logM
        df.loc[idx, 'logMBH_CIV_err']    = logM_err

    # Write output
    df.to_csv(output_csv, index=False)
    print(f"Saved derived quantities to {output_csv}")


# if __name__ == "__main__":
#     input_csv  = base_path + "Target_lists/target_list_info_for_calc.csv"
#     output_csv = base_path + "Target_lists/civ_output_with_derived.csv"

#     main(input_csv, output_csv)

f = fits.open("/Users/nayera/PyQSOFit/CIV_galaxy_spectra_targets/J1245+4348/output/C_01spec-6617-56365-0860.fits")
print(f[1].data['L1350'], f[1].data['L1450'], f[1].data['PL_slope'])
