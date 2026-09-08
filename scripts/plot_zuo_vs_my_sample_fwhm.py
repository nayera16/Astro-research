import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LN10 = np.log(10.0)
RICHARDS_L1350_BOL_CORRECTION = 3.81

# Published H-beta black hole masses from Zuo et al. Table 2, column 6.
ZUO_LOG_MBH_HB = {
    "J011521.20+152453.3": 9.61,
    "J014214.75+002324.2": 9.58,
    "J015741.57-010629.6": 9.92,
    "J025021.76-075749.9": 9.44,
    "J025905.63+001121.9": 9.72,
    "J030341.04-002321.9": 9.28,
    "J030449.85-000813.4": 9.38,
    "J075303.34+423130.8": 10.14,
    "J080430.56+542041.1": 9.55,
    "J080819.69+373047.3": 10.05,
    "J080956.02+502000.9": 9.81,
    "J081855.77+095848.0": 9.77,
    "J090033.50+421547.0": 9.63,
    "J094202.04+042244.5": 9.71,
    "J115954.33+201921.1": 10.05,
    "J173352.23+540030.4": 9.51,
    "J213023.61+122252.0": 9.54,
    "J224956.08+000218.0": 9.42,
    "J230301.45-093930.7": 9.79,
}

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

SCRIPT_DIR = Path(__file__).resolve().parent
MY_CSV = Path("/Users/nayera/PyQSOFit/Target_lists/civ_output_with_derived.csv")
ZUO_CSV = Path("/Users/nayera/PyQSOFit/Target_lists/zuo_civ_hbeta_fwhm_ratio_comparison.csv")
OUTPUT_PLOT = SCRIPT_DIR / "logFWHM_Hb_vs_logFWHM_CIV_my_sample_vs_zuo_blueshift_corrected.png"
OUTPUT_LBOL_PLOT = SCRIPT_DIR / "logLbol_my_sample_vs_zuo.png"
OUTPUT_HB_MASS_PLOT = SCRIPT_DIR / "logMBH_Hb_my_sample_vs_zuo.png"


def log_values_with_errors(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = np.isfinite(values) & np.isfinite(errors) & (values > 0) & (errors >= 0)
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])

    valid_values = values[mask]
    valid_errors = errors[mask]
    x = np.log10(valid_values)
    xerr = valid_errors / (LN10 * valid_values)
    return x, xerr, np.arange(len(valid_values))


def load_my_sample():
    df = pd.read_csv(MY_CSV)
    mask = (
        np.isfinite(df["FWHM_CIV_corr_blueshift"]) &
        np.isfinite(df["fwhm_hb"]) &
        np.isfinite(df["FWHM_err_mcmc"]) &
        np.isfinite(df["fwhm_hb_err"]) &
        (df["FWHM_CIV_corr_blueshift"] > 0) &
        (df["fwhm_hb"] > 0)
    )
    civ = df.loc[mask, "FWHM_CIV_corr_blueshift"].to_numpy()
    hb = df.loc[mask, "fwhm_hb"].to_numpy()
    civ_err = df.loc[mask, "FWHM_err_mcmc"].to_numpy()
    hb_err = df.loc[mask, "fwhm_hb_err"].to_numpy()

    x, xerr, _ = log_values_with_errors(civ, civ_err)
    y, yerr, _ = log_values_with_errors(hb, hb_err)
    return x, y, xerr, yerr


def load_zuo_sample():
    df = pd.read_csv(ZUO_CSV)
    mask = (
        np.isfinite(df["civ_fwhm_km_s"]) &
        np.isfinite(df["zuo_hbeta_fwhm_km_s"]) &
        np.isfinite(df["civ_fwhm_err_km_s"]) &
        np.isfinite(df["zuo_hbeta_fwhm_err_km_s"]) &
        (df["civ_fwhm_km_s"] > 0) &
        (df["zuo_hbeta_fwhm_km_s"] > 0)
    )
    civ = df.loc[mask, "civ_fwhm_km_s"].to_numpy()
    hb = df.loc[mask, "zuo_hbeta_fwhm_km_s"].to_numpy()
    civ_err = df.loc[mask, "civ_fwhm_err_km_s"].to_numpy()
    hb_err = df.loc[mask, "zuo_hbeta_fwhm_err_km_s"].to_numpy()

    x, xerr, _ = log_values_with_errors(civ, civ_err)
    y, yerr, _ = log_values_with_errors(hb, hb_err)
    return x, y, xerr, yerr


def parse_table_error(series):
    return pd.to_numeric(
        series.astype(str).str.replace("<", "", regex=False),
        errors="coerce"
    )


def load_my_lbol_sample():
    df = pd.read_csv(MY_CSV)
    log_l1350 = pd.to_numeric(df["logL1350"], errors="coerce")
    log_l1350_err = parse_table_error(df["logL1350_err"])
    log_lbol = log_l1350 + np.log10(RICHARDS_L1350_BOL_CORRECTION)

    mask = (
        np.isfinite(log_lbol) &
        np.isfinite(log_l1350_err) &
        (log_lbol > 10)
    )
    return log_lbol.loc[mask].to_numpy(), log_l1350_err.loc[mask].to_numpy()


def load_zuo_lbol_sample():
    df = pd.read_csv(ZUO_CSV)
    log_lbol = pd.to_numeric(df["zuo_log_lbol_erg_s"], errors="coerce")
    log_lbol_err = parse_table_error(df["zuo_log_lbol_err"])

    mask = (
        np.isfinite(log_lbol) &
        np.isfinite(log_lbol_err) &
        (log_lbol > 10)
    )
    return log_lbol.loc[mask].to_numpy(), log_lbol_err.loc[mask].to_numpy()


def load_my_hb_mass_sample():
    df = pd.read_csv(MY_CSV)
    log_mbh_hb = pd.to_numeric(df["logMBH_Hb"], errors="coerce")
    return log_mbh_hb.loc[np.isfinite(log_mbh_hb)].to_numpy()


def load_zuo_hb_mass_sample():
    """Load the published Zuo et al. H-beta masses (Table 2, column 6)."""
    df = pd.read_csv(ZUO_CSV)
    log_mbh_hb = df["name_sdss"].map(ZUO_LOG_MBH_HB)

    missing = df.loc[log_mbh_hb.isna(), "name_sdss"].tolist()
    if missing:
        raise ValueError(f"Missing published H-beta masses for Zuo targets: {missing}")

    return log_mbh_hb.to_numpy(dtype=float)


def plot_hb_mass_comparison():
    log_mbh_my = load_my_hb_mass_sample()
    log_mbh_zuo = load_zuo_hb_mass_sample()

    fig, ax = plt.subplots(figsize=(6, 5))

    all_masses = np.concatenate([log_mbh_my, log_mbh_zuo])
    bin_min = np.floor(all_masses.min() * 5) / 5
    bin_max = np.ceil(all_masses.max() * 5) / 5
    bins = np.arange(bin_min, bin_max + 0.2, 0.2)

    ax.hist(
        log_mbh_my,
        bins=bins,
        histtype="stepfilled",
        alpha=0.35,
        color="tab:blue",
        edgecolor="tab:blue",
        linewidth=1.5,
        density=True,
        label=fr"My sample ($N={len(log_mbh_my)}$)"
    )
    ax.hist(
        log_mbh_zuo,
        bins=bins,
        histtype="step",
        color="tab:orange",
        linewidth=1.8,
        density=True,
        label=fr"Zuo sample ($N={len(log_mbh_zuo)}$)"
    )

    ax.axvline(np.median(log_mbh_my), color="tab:blue", linestyle="--", linewidth=1.4)
    ax.axvline(np.median(log_mbh_zuo), color="tab:orange", linestyle="--", linewidth=1.4)

    ax.set_xlabel(r"$\log\,M_{\rm BH}\ {\rm (H\beta)}\ (M_\odot)$")
    ax.set_ylabel("Normalized count")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_HB_MASS_PLOT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_HB_MASS_PLOT}")


def plot_lbol_comparison():
    log_lbol_my, _ = load_my_lbol_sample()
    log_lbol_zuo, _ = load_zuo_lbol_sample()

    fig, ax = plt.subplots(figsize=(6, 5))

    all_lbol = np.concatenate([log_lbol_my, log_lbol_zuo])
    bin_min = np.floor(all_lbol.min() * 5) / 5
    bin_max = np.ceil(all_lbol.max() * 5) / 5
    bins = np.arange(bin_min, bin_max + 0.2, 0.2)

    ax.hist(
        log_lbol_my,
        bins=bins,
        histtype="stepfilled",
        alpha=0.35,
        color="tab:blue",
        edgecolor="tab:blue",
        linewidth=1.5,
        density=True,
        label=fr"My sample ($N={len(log_lbol_my)}$)"
    )
    ax.hist(
        log_lbol_zuo,
        bins=bins,
        histtype="step",
        color="tab:orange",
        linewidth=1.8,
        density=True,
        label=fr"Zuo sample ($N={len(log_lbol_zuo)}$)"
    )

    ax.axvline(np.median(log_lbol_my), color="tab:blue", linestyle="--", linewidth=1.4)
    ax.axvline(np.median(log_lbol_zuo), color="tab:orange", linestyle="--", linewidth=1.4)

    ax.set_xlabel(r"$\log\,L_{\rm bol}\ (\mathrm{erg\ s^{-1}})$")
    ax.set_ylabel("Normalized count")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_LBOL_PLOT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_LBOL_PLOT}")


def main():
    x_my, y_my, xerr_my, yerr_my = load_my_sample()
    x_zuo, y_zuo, xerr_zuo, yerr_zuo = load_zuo_sample()

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.errorbar(
        x_my, y_my,
        xerr=xerr_my, yerr=yerr_my,
        fmt="o", ms=7,
        mfc="tab:blue", mec="tab:blue",
        ecolor="tab:blue", elinewidth=1.2, capsize=3,
        linestyle="none",
        label="My sample"
    )

    ax.errorbar(
        x_zuo, y_zuo,
        xerr=xerr_zuo, yerr=yerr_zuo,
        fmt="D", ms=7,
        mfc="none", mec="tab:orange",
        ecolor="tab:orange", elinewidth=1.2, capsize=3,
        linestyle="none",
        label="Zuo sample"
    )

    ax.set_xlabel(r"$\log\,\mathrm{FWHM}_{\rm C\,IV}\ (\mathrm{km\ s^{-1}})$")
    ax.set_ylabel(r"$\log\,\mathrm{FWHM}_{\rm H\beta}\ (\mathrm{km\ s^{-1}})$")
    ax.set_xlim(3.2, 4.2)
    ax.set_ylim(3.2, 3.8)

    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PLOT}")

    plot_lbol_comparison()
    plot_hb_mass_comparison()


if __name__ == "__main__":
    main()
