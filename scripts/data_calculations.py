from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.constants import c

# sys.path.append('../')

LAM_LAB_CIV = 1549.06    # in angstroms, rest wavelength of C IV
C_KMS = c.to('km/s').value
LN10 = np.log(10.0)

# Zuo+2020 calibration constants
ALPHA_BS = 0.67             # from Zuo2020 Eq. 6 (blueshift relation)
BETA_BS = 0.41
ALPHA_AS = 2.03             # from Zuo2020 Eq. 7 (asymmetry relation)
BETA_AS = -1.1

# Vestergaard & Peterson (2006) mass calibration constants
A = 0.66
B = 0.53
C = 2.0

base_path = "/Users/nayera/PyQSOFit/" # local directory with all my data inputs/outputs
BASE_PATH = Path(base_path)
TARGET_LIST_INFO_CSV = BASE_PATH / "Target_lists" / "target_list_info_for_calc.csv"

def get_L1350_from_fits(fits_path):
    """
    Get input info from FITS file and return L1350 and its uncertainty. PyQSOFit did not 
    cover 1350 range so we could not direclty output L1350 but needed it in the Vestergaard 
    and Peterson (2006) mass estimate equation. 

    Compute log10(L1350/erg s^-1) using:
        logL1350 = logL1450 + (slope + 1) * log10(1350/1450)

    Returns logL1350 and propagated error.
    """
    with fits.open(fits_path) as f:
        tab = f[1].data  
        row = tab[0]

        # Get L1450 and power law slope
        logL1450 = float(row['L1450'])
        logL1450_err = float(row['L1450_err'])
        slope = float(row['PL_slope'])
        slope_err = float(row['PL_slope_err'])

        # calculate logL1350
        log_ratio = np.log10(1350.0 / 1450.0)
        logL1350 = logL1450 + (slope + 1.0) * log_ratio

        # Error propagation in log-space
        logL1350_err = np.sqrt(logL1450_err**2 + (log_ratio * slope_err)**2)

    return logL1350, logL1350_err
    

def blueshift_calc(lam_half, lam_lab=LAM_LAB_CIV):
    # Compute C IV blueshift (km/s) using the half flux peak wavelength (rest frame)
    return C_KMS * (lam_lab - lam_half) / lam_lab


def asym_calc(lam_blue, lam_red, lam_peak):
    """
    Use Zuo+2020 asymmetry calculation:
        A_S = ln(lam_red / lam_peak) / ln(lam_blue / lam_peak)

    Returns np.nan if the input is invalid.
    """
    if lam_blue <= 0 or lam_red <= 0 or lam_peak <= 0:
        return np.nan
    num = np.log(lam_red / lam_peak)
    den = np.log(lam_blue / lam_peak)
    if den == 0:
        return np.nan
    return num / den

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
    Propagate uncertainties for BHM assuming
    uncorrelated errors in L1350 and FWHM.
        sigma_M^2 = (0.53*sigma_logL)^2 + (2/(ln(10)*FWHM)*sigma_FWHM)^2
    """
    term_L = B * logL1350_err
    term_F = (C / (LN10 * FWHM)) * FWHM_err

    sigma_stat = np.sqrt(term_L**2 + term_F**2)

    return sigma_stat

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

def resolve_fit_output_path(row):
    """
    Return the PyQSOFit output FITS path for either the old calculation input
    format or the 1600 batch output format.
    """
    target = row["Target"]
    fits_value = str(row["fits_path"])
    model = row.get("Model", np.nan)
    file_name = row.get("file_name", np.nan)

    target_dir = BASE_PATH / "CIV_galaxy_spectra_targets" / target
    output_dir = target_dir / "output"

    if pd.notna(model) and pd.notna(file_name):
        return output_dir / f"{model}_{Path(str(file_name)).name}"

    fits_path = Path(fits_value)
    if fits_path.suffix == ".fits":
        if fits_path.parent.name == "output":
            return fits_path
        return output_dir / fits_path.name

    return output_dir / f"{fits_value}.fits"

def is_valid_number(value):
    return pd.notna(value) and np.isfinite(value)


def add_hbeta_target_values(df):
    """
    Add the H-beta comparison values from the target-info table for targets that
    have them. The final C IV target CSV can include newer C IV-only targets, so
    unmatched rows are kept with NaN H-beta values.
    """
    if not TARGET_LIST_INFO_CSV.exists():
        print(f"Missing H-beta target info file: {TARGET_LIST_INFO_CSV}")
        return df

    hbeta_cols = ["Target", "logMBH_Hb", "logMBH_Hb_err", "fwhm_hb", "fwhm_hb_err"]
    hbeta_df = pd.read_csv(TARGET_LIST_INFO_CSV, encoding="utf-8-sig")
    available_cols = [col for col in hbeta_cols if col in hbeta_df.columns]
    missing_cols = sorted(set(hbeta_cols) - set(available_cols))
    if missing_cols:
        print(f"Missing H-beta columns in target info file: {', '.join(missing_cols)}")
        return df

    hbeta_df = hbeta_df[hbeta_cols].drop_duplicates(subset=["Target"])
    merged = df.merge(hbeta_df, on="Target", how="left", suffixes=("", "_target_info"))

    for col in hbeta_cols:
        if col == "Target":
            continue
        source_col = f"{col}_target_info"
        if source_col in merged.columns:
            if col in df.columns:
                merged[col] = merged[col].combine_first(merged[source_col])
            else:
                merged[col] = merged[source_col]
            merged.drop(columns=[source_col], inplace=True)

    return merged

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = add_hbeta_target_values(df)

    params = [ 
        'logL1350', 'logL1350_err', 'CIV_blueshift_kms', 
        'CIV_asym', 'logMBH_CIV', 'logMBH_CIV_err', 
        'FWHM_CIV_corr_blueshift','logMBH_CIV_corr_blueshift',
        'FWHM_CIV_corr_asym','logMBH_CIV_corr_asym', "log_mass_ratio_CIV_Hb"
    ]

    for param in params:
        df[param] = np.nan

    for idx, row in df.iterrows():
        fits_path = resolve_fit_output_path(row)
        if not fits_path.exists():
            print(f"Missing FITS output for row {idx}: {fits_path}")
            continue

        try:
            logL1350, logL1350_err = get_L1350_from_fits(fits_path)
        except Exception as exc:
            print(f"Could not read L1350 for row {idx} from {fits_path}: {exc}")
            continue

        lam_blue = row['lam_blue']
        lam_red = row['lam_red']
        lam_half = row['lam_half']
        lam_peak = row['br_peak']          
        FWHM = row['FWHM_CIV']
        FWHM_err = row['FWHM_err_mcmc']

        if not all(is_valid_number(value) for value in [lam_blue, lam_red, lam_half, lam_peak, FWHM, FWHM_err]):
            df.loc[idx, 'logL1350'] = logL1350
            df.loc[idx, 'logL1350_err'] = logL1350_err
            continue

        blueshift = blueshift_calc(lam_half)
        asym = asym_calc(lam_blue, lam_red, lam_peak)
        logM = bhm_civ_calc(logL1350, FWHM)
        logM_err = bhm_civ_error(logL1350_err, FWHM, FWHM_err)
        FWHM_corr_bs = fwhm_blueshift_corrected(FWHM, blueshift)
        logM_corr_bs = bhm_civ_corrected(logL1350, FWHM_corr_bs)
        FWHM_corr_as = fwhm_asym_corrected(FWHM, asym)
        logM_corr_as = bhm_civ_corrected(logL1350, FWHM_corr_as)

        df.loc[idx, 'logL1350']  = logL1350
        df.loc[idx, 'logL1350_err']  = logL1350_err
        df.loc[idx, 'CIV_blueshift_kms'] = blueshift
        df.loc[idx, 'CIV_asym'] = asym
        df.loc[idx, 'logMBH_CIV'] = logM
        df.loc[idx, 'logMBH_CIV_err'] = logM_err
        df.loc[idx, 'FWHM_CIV_corr_blueshift'] = FWHM_corr_bs
        df.loc[idx, 'logMBH_CIV_corr_blueshift'] = logM_corr_bs
        df.loc[idx, 'FWHM_CIV_corr_asym'] = FWHM_corr_as
        df.loc[idx, 'logMBH_CIV_corr_asym'] = logM_corr_as
        if (
            "logMBH_Hb" in df.columns
            and np.isfinite(df.loc[idx, "logMBH_CIV"])
            and np.isfinite(df.loc[idx, "logMBH_Hb"])
        ):
            df.loc[idx, "log_mass_ratio_CIV_Hb"] = (
                df.loc[idx, "logMBH_CIV"] - df.loc[idx, "logMBH_Hb"]
            )

    # Replace -1 or empty cells with NaN before saving
    bad_cols = [col for col in df.select_dtypes(include=[np.number]).columns]
    df[bad_cols] = df[bad_cols].where(df[bad_cols] != -1, other=np.nan)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = base_path + "Target_lists/civ_output_final_target_values.csv"
    output_csv = base_path + "Target_lists/civ_output_final_target_values.csv"
    main(input_csv, output_csv)
