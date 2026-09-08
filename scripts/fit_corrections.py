import numpy as np
import pandas as pd
import linmix

# paths 
INPUT_CSV  = "/Users/nayera/PyQSOFit/Target_lists/civ_output_final_target_values.csv"
OUTPUT_CSV = "/Users/nayera/PyQSOFit/Target_lists/civ_output_final_target_values.csv"

LN10 = np.log(10.0)

# Zuo+2020 calibration constants for comparison
ALPHA_BS_ZUO = 0.67
BETA_BS_ZUO  = 0.41
ALPHA_AS_ZUO = 2.03
BETA_AS_ZUO  = -1.1

# vestergaard and peterson 2006 mass calibration constants
A = 0.66
B = 0.53
C = 2.0

def run_linmix(x, y, xsig, ysig, seed=42):
    """
    Run linmix (Kelly 2007) Bayesian linear regression to account for x and y errors.
    Returns (alpha, alpha_err, beta, beta_err) as medians and 1-sigma (16th-84th 
    percentile half-width).
    """
    lm = linmix.LinMix(x, y, xsig=xsig, ysig=ysig, seed=seed)
    lm.run_mcmc(silent=True)

    alpha_samples = lm.chain["alpha"]
    beta_samples = lm.chain["beta"]

    alpha = np.median(alpha_samples)
    alpha_err = (np.percentile(alpha_samples, 84) - np.percentile(alpha_samples, 16)) / 2.0
    beta = np.median(beta_samples)
    beta_err = (np.percentile(beta_samples, 84) - np.percentile(beta_samples, 16)) / 2.0

    return alpha, alpha_err, beta, beta_err

# recalculate using new alpha, beta from linmix

def fwhm_blueshift_corrected_fitted(FWHM, blueshift, alpha, beta):
    if blueshift <= 0 or np.isnan(blueshift):
        return np.nan
    denom = alpha + beta * (blueshift / 1000.0)
    if denom <= 0:
        return np.nan
    return FWHM / denom


def fwhm_asym_corrected_fitted(FWHM, asym, alpha, beta):
    if np.isnan(asym):
        return np.nan
    denom = alpha + beta * asym
    if denom <= 0:
        return np.nan
    return FWHM / denom


def bhm_civ(logL1350, FWHM):
    return A + B * (logL1350 - 44.0) + C * np.log10(FWHM)

# Perform fitting to extract alpha and beta
def main():
    df = pd.read_csv(INPUT_CSV)

    # fitting inputs
    fit_mask = (
        np.isfinite(df["FWHM_CIV"]) &
        np.isfinite(df["FWHM_err_mcmc"]) &
        np.isfinite(df["fwhm_hb"]) &
        np.isfinite(df["fwhm_hb_err"]) &
        np.isfinite(df["CIV_blueshift_kms"]) &
        np.isfinite(df["CIV_asym"]) &
        np.isfinite(df["CIV_asym_err_mcmc"]) &
        (df["CIV_asym_err_mcmc"] > 0) &
        (df["CIV_blueshift_kms"] > 0)
    )
    dff = df[fit_mask].copy()
    print(f"Objects used for fitting: {len(dff)}")

    # y-axis for both fits is FWHM_CIV / FWHM_Hb
    ratio = dff["FWHM_CIV"].to_numpy() / dff["fwhm_hb"].to_numpy()

    # propagate error
    ratio_err  = ratio * np.sqrt(
        (dff["FWHM_err_mcmc"].to_numpy() / dff["FWHM_CIV"].to_numpy())**2 +
        (dff["fwhm_hb_err"].to_numpy() / dff["fwhm_hb"].to_numpy())**2
    )

    # blueshift fit equation x-axis: blueshift / 1000 km/s
    x_bs = dff["CIV_blueshift_kms"].to_numpy() / 1000.0
    xsig_bs = np.full_like(x_bs, 1e-6)   # negligible

    # asymmetry fit equation x-axis: AS_CIV 
    x_as = dff["CIV_asym"].to_numpy()
    xsig_as = dff["CIV_asym_err_mcmc"].to_numpy()

    # Run linmix fits 
    alpha_bs, alpha_bs_err, beta_bs, beta_bs_err = run_linmix(x_bs, ratio, xsig_bs, ratio_err )
    alpha_as, alpha_as_err, beta_as, beta_as_err = run_linmix(x_as, ratio, xsig_as, ratio_err)

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Parameter':<30} {'Zuo+2020':>15} {'This work':>15}")
    print("-" * 65)
    print(f"{'Eq.6  alpha (blueshift)':<30} "
          f"{ALPHA_BS_ZUO:>8.2f}          "
          f"{alpha_bs:>6.2f} ± {alpha_bs_err:.2f}")
    print(f"{'Eq.6  beta  (blueshift)':<30} "
          f"{BETA_BS_ZUO:>8.2f}          "
          f"{beta_bs:>6.2f} ± {beta_bs_err:.2f}")
    print(f"{'Eq.7  alpha (asymmetry)':<30} "
          f"{ALPHA_AS_ZUO:>8.2f}          "
          f"{alpha_as:>6.2f} ± {alpha_as_err:.2f}")
    print(f"{'Eq.7  beta  (asymmetry)':<30} "
          f"{BETA_AS_ZUO:>8.2f}          "
          f"{beta_as:>6.2f} ± {beta_as_err:.2f}")
    print("=" * 65)

    # apply corrections and add to output dataframe
    new_cols = [
        "FWHM_CIV_fitted_corr_blueshift",
        "logMBH_CIV_fitted_corr_blueshift",
        "FWHM_CIV_fitted_corr_asym",
        "logMBH_CIV_fitted_corr_asym",
    ]
    for col in new_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        logL1350 = row["logL1350"]
        FWHM = row["FWHM_CIV"]
        bs = row["CIV_blueshift_kms"]
        asym = row["CIV_asym"]

        if not (np.isfinite(logL1350) and np.isfinite(FWHM)):
            continue

        # blueshift correction with fitted params
        FWHM_bs = fwhm_blueshift_corrected_fitted(FWHM, bs, alpha_bs, beta_bs)
        if np.isfinite(FWHM_bs) and FWHM_bs > 0:
            df.loc[idx, "FWHM_CIV_fitted_corr_blueshift"] = FWHM_bs
            df.loc[idx, "logMBH_CIV_fitted_corr_blueshift"]= bhm_civ(logL1350, FWHM_bs)

        # asymmetry correction with fitted params
        FWHM_as = fwhm_asym_corrected_fitted(FWHM, asym, alpha_as, beta_as)
        if np.isfinite(FWHM_as) and FWHM_as > 0:
            df.loc[idx, "FWHM_CIV_fitted_corr_asym"] = FWHM_as
            df.loc[idx, "logMBH_CIV_fitted_corr_asym"] = bhm_civ(logL1350, FWHM_as)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved fitted corrections to {OUTPUT_CSV}")

    # Scatter statistics for new corrections
    def inner50_scatter(mass_civ, mass_hb):
        diff = mass_civ - mass_hb
        median_val = np.median(diff)
        scatter = (np.percentile(diff, 75) - np.percentile(diff, 25)) / 2.0
        return median_val, scatter

    # mask for objects with all six mass columns finite
    plot_mask = (
        np.isfinite(df["logMBH_CIV"]) &
        np.isfinite(df["logMBH_CIV_corr_blueshift"]) &
        np.isfinite(df["logMBH_CIV_corr_asym"]) &
        np.isfinite(df["logMBH_CIV_fitted_corr_blueshift"]) &
        np.isfinite(df["logMBH_CIV_fitted_corr_asym"]) &
        np.isfinite(df["logMBH_Hb"])
    )
    dfp = df[plot_mask].copy()
    print(f"\nObjects in scatter comparison: {len(dfp)}")

    hb = dfp["logMBH_Hb"].to_numpy()

    med_raw, scat_raw  = inner50_scatter(dfp["logMBH_CIV"].to_numpy(), hb)
    med_bs_zuo, scat_bs_zuo = inner50_scatter(dfp["logMBH_CIV_corr_blueshift"].to_numpy(), hb)
    med_as_zuo, scat_as_zuo = inner50_scatter(dfp["logMBH_CIV_corr_asym"].to_numpy(),  hb)
    med_bs_fit, scat_bs_fit = inner50_scatter(dfp["logMBH_CIV_fitted_corr_blueshift"].to_numpy(), hb)
    med_as_fit, scat_as_fit = inner50_scatter(dfp["logMBH_CIV_fitted_corr_asym"].to_numpy(), hb)

    print("\n" + "=" * 65)
    print(f"{'Correction':<35} {'Median':>8} {'Scatter':>10} {'Δσ':>8}")
    print("-" * 65)
    print(f"{'Uncorrected':<35} {med_raw:>+8.3f} {scat_raw:>10.3f} {'—':>8}")
    print(f"{'Blueshift corr. (Zuo alpha,beta)':<35} {med_bs_zuo:>+8.3f} {scat_bs_zuo:>10.3f} {scat_raw-scat_bs_zuo:>+8.3f}")
    print(f"{'Asymmetry corr. (Zuo alpha,beta)':<35} {med_as_zuo:>+8.3f} {scat_as_zuo:>10.3f} {scat_raw-scat_as_zuo:>+8.3f}")
    print(f"{'Blueshift corr. (fitted alpha,beta)':<35} {med_bs_fit:>+8.3f} {scat_bs_fit:>10.3f} {scat_raw-scat_bs_fit:>+8.3f}")
    print(f"{'Asymmetry corr. (fitted alpha,beta)':<35} {med_as_fit:>+8.3f} {scat_as_fit:>10.3f} {scat_raw-scat_as_fit:>+8.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
