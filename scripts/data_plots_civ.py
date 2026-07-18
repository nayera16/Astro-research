from pathlib import Path
import os

MPLCONFIGDIR = Path("1600 data outputs") / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LN10 = np.log(10.0)

CSV_PATH = Path("/Users/nayera/PyQSOFit/Target_lists/civ_output_final_target_values.csv")
OUTPUT_DIR = Path("1600 data outputs")
OUTPUT_TAG = "1600"


# set plot style
plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in"
})


def output_path(filename):
    return OUTPUT_DIR / filename


def save_figure(fig, filename):
    path = output_path(filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


def has_columns(df, columns, plot_name):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"Skipping {plot_name}: missing columns: {', '.join(missing)}")
        return False
    return True


def finite_positive(series):
    return np.isfinite(series) & (series > 0)


def median_and_inner50_scatter(mass_civ, mass_hb):
    diff = mass_civ - mass_hb
    median_val = np.median(diff)
    scatter = (np.percentile(diff, 75) - np.percentile(diff, 25)) / 2.0
    return median_val, scatter


def plot_blueshift_hist(df):
    plot_name = "C IV blueshift histogram"
    if not has_columns(df, ["CIV_blueshift_kms"], plot_name):
        return

    dfh = df[np.isfinite(df["CIV_blueshift_kms"])].copy()
    if dfh.empty:
        print(f"Skipping {plot_name}: no finite CIV_blueshift_kms values")
        return

    v = dfh["CIV_blueshift_kms"].to_numpy()
    vmin = np.floor(v.min() / 500) * 500
    vmax = np.ceil(v.max() / 500) * 500
    bins = np.arange(vmin, vmax + 500, 500)

    hist = plt.figure(figsize=(6, 5))
    ax1 = hist.add_subplot(111)
    m = max(abs(vmin), abs(vmax))
    ax1.set_xlim(-m, m)
    ax1.hist(v, bins=bins, color="lightblue", edgecolor="black", linewidth=1.2)
    ax1.set_xlabel(r"$\Delta V_{\rm C\,IV}$  (km s$^{-1}$)")
    ax1.set_ylabel("Number of quasars")

    plt.tight_layout()
    save_figure(hist, f"blueshift_hist_all_{OUTPUT_TAG}.png")
    plt.close(hist)


def plot_fwhm_vs_blueshift(df):
    plot_name = "FWHM vs C IV blueshift"
    if not has_columns(df, ["CIV_blueshift_kms", "FWHM_CIV"], plot_name):
        return

    dff = df[np.isfinite(df["CIV_blueshift_kms"]) & finite_positive(df["FWHM_CIV"])].copy()
    if dff.empty:
        print(f"Skipping {plot_name}: no rows with finite blueshift and positive FWHM_CIV")
        return

    fig = plt.figure(figsize=(6, 5))
    ax2 = fig.add_subplot(111)
    ax2.scatter(
        dff["CIV_blueshift_kms"],
        dff["FWHM_CIV"],
        facecolors="none",
        edgecolors="black",
        linewidth=1.2,
        s=70
    )

    ax2.set_xlabel(r"$\Delta V_{\rm C\,IV}$  (km s$^{-1}$)")
    ax2.set_ylabel(r"FWHM$_{\rm C\,IV}$  (km s$^{-1}$)")
    ax2.set_xlim(0, 5000)
    ax2.set_yscale("log")

    plt.tight_layout()
    save_figure(fig, f"fwhm_vs_blueshift_{OUTPUT_TAG}.png")
    plt.close(fig)


def plot_log_fwhm_all_corrections(df):
    plot_name = "log FWHM_Hbeta vs log FWHM_CIV"
    required = [
        "FWHM_CIV", "FWHM_err_mcmc", "FWHM_CIV_corr_blueshift",
        "FWHM_CIV_corr_asym", "fwhm_hb", "fwhm_hb_err"
    ]
    if not has_columns(df, required, plot_name):
        return

    dff = df[
        finite_positive(df["FWHM_CIV"]) &
        np.isfinite(df["FWHM_err_mcmc"]) &
        finite_positive(df["FWHM_CIV_corr_blueshift"]) &
        finite_positive(df["FWHM_CIV_corr_asym"]) &
        finite_positive(df["fwhm_hb"]) &
        np.isfinite(df["fwhm_hb_err"])
    ].copy()
    if dff.empty:
        print(f"Skipping {plot_name}: no rows with all required finite values")
        return

    x = np.log10(dff["FWHM_CIV"].to_numpy())
    x_bs = np.log10(dff["FWHM_CIV_corr_blueshift"].to_numpy())
    x_as = np.log10(dff["FWHM_CIV_corr_asym"].to_numpy())
    y = np.log10(dff["fwhm_hb"].to_numpy())

    xerr = dff["FWHM_err_mcmc"].to_numpy() / (LN10 * dff["FWHM_CIV"].to_numpy())
    xerr_bs = dff["FWHM_err_mcmc"].to_numpy() / (LN10 * dff["FWHM_CIV"].to_numpy())
    xerr_as = dff["FWHM_err_mcmc"].to_numpy() / (LN10 * dff["FWHM_CIV"].to_numpy())
    yerr = dff["fwhm_hb_err"].to_numpy() / (LN10 * dff["fwhm_hb"].to_numpy())

    fig_main = plt.figure(figsize=(6, 5))
    ax3 = fig_main.add_subplot(111)

    ax3.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="o", ms=7,
        mfc="black", mec="black", ecolor="black", elinewidth=1.2, capsize=3,
        linestyle="none", label="Uncorrected C IV"
    )
    ax3.errorbar(
        x_bs, y, xerr=xerr_bs, yerr=yerr, fmt="D", ms=7,
        mfc="none", mec="red", ecolor="red", elinewidth=1.2, capsize=3,
        linestyle="none", label="Blueshift corrected"
    )
    ax3.errorbar(
        x_as, y, xerr=xerr_as, yerr=yerr, fmt="s", ms=7,
        mfc="none", mec="blue", ecolor="blue", elinewidth=1.2, capsize=3,
        linestyle="none", label="Asymmetry corrected"
    )

    ax3.set_xlabel(r"$\log\,\mathrm{FWHM}_{\rm C\,IV}\ (\mathrm{km\ s^{-1}})$")
    ax3.set_ylabel(r"$\log\,\mathrm{FWHM}_{\rm H\beta}\ (\mathrm{km\ s^{-1}})$")
    ax3.set_xlim(3.2, 4.0)
    ax3.set_ylim(3.2, 3.8)

    xmin, xmax = ax3.get_xlim()
    line = np.linspace(xmin, xmax, 100)
    ax3.plot(line, line, linestyle=":", color="black", linewidth=1.2, label="1:1 line")
    ax3.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    save_figure(fig_main, f"logFWHM_Hb_vs_logFWHM_CIV_all_corrections_{OUTPUT_TAG}.png")
    plt.close(fig_main)


def plot_log_mbh_all_corrections(df):
    plot_name = "log M_BH(Hbeta) vs log M_BH(CIV)"
    required = [
        "logMBH_CIV", "logMBH_CIV_err", "logMBH_CIV_corr_blueshift",
        "logMBH_CIV_corr_asym", "logMBH_Hb", "logMBH_Hb_err"
    ]
    if not has_columns(df, required, plot_name):
        return

    dff = df[
        np.isfinite(df["logMBH_CIV"]) &
        (df["logMBH_CIV_err"] > 0) &
        np.isfinite(df["logMBH_CIV_corr_blueshift"]) &
        np.isfinite(df["logMBH_CIV_corr_asym"]) &
        np.isfinite(df["logMBH_Hb"]) &
        (df["logMBH_Hb_err"] > 0) &
        np.isfinite(df["logMBH_Hb_err"])
    ].copy()
    if dff.empty:
        print(f"Skipping {plot_name}: no rows with all required finite values")
        return

    x = dff["logMBH_CIV"].to_numpy()
    x_bs = dff["logMBH_CIV_corr_blueshift"].to_numpy()
    x_as = dff["logMBH_CIV_corr_asym"].to_numpy()
    y = dff["logMBH_Hb"].to_numpy()
    xerr = dff["logMBH_CIV_err"].to_numpy()
    yerr = dff["logMBH_Hb_err"].to_numpy()

    med_raw, scat_raw = median_and_inner50_scatter(x, y)
    med_bs, scat_bs = median_and_inner50_scatter(x_bs, y)
    med_as, scat_as = median_and_inner50_scatter(x_as, y)

    print("=" * 55)
    print(f"{'Correction':<25} {'Median (dex)':>12} {'Scatter (dex)':>14}")
    print("-" * 55)
    print(f"{'Uncorrected':<25} {med_raw:>+12.3f} {scat_raw:>14.3f}")
    print(f"{'Blueshift corrected':<25} {med_bs:>+12.3f} {scat_bs:>14.3f}")
    print(f"{'Asymmetry corrected':<25} {med_as:>+12.3f} {scat_as:>14.3f}")
    print("=" * 55)
    print(f"Scatter reduction (blueshift): {scat_raw - scat_bs:+.3f} dex")
    print(f"Scatter reduction (asymmetry): {scat_raw - scat_as:+.3f} dex")
    print(f"N objects (all three valid):   {len(dff)}")

    fig_main_bhm = plt.figure(figsize=(6, 5))
    ax4 = fig_main_bhm.add_subplot(111)
    ax4.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="o", ms=7,
        mfc="black", mec="black", ecolor="black", elinewidth=1.2, capsize=3,
        linestyle="none",
        label=fr"Uncorrected  ($\Delta={med_raw:+.2f}$, $\sigma={scat_raw:.2f}$)"
    )
    ax4.errorbar(
        x_bs, y, xerr=xerr, yerr=yerr, fmt="D", ms=7,
        mfc="none", mec="red", ecolor="red", elinewidth=1.2, capsize=3,
        linestyle="none",
        label=fr"Blueshift corr.  ($\Delta={med_bs:+.2f}$, $\sigma={scat_bs:.2f}$)"
    )
    ax4.errorbar(
        x_as, y, xerr=xerr, yerr=yerr, fmt="s", ms=7,
        mfc="none", mec="blue", ecolor="blue", elinewidth=1.2, capsize=3,
        linestyle="none",
        label=fr"Asymmetry corr.  ($\Delta={med_as:+.2f}$, $\sigma={scat_as:.2f}$)"
    )

    ax4.set_xlabel(r"$\log\,M_{\rm BH}\ {\rm (C\,IV)}\ (M_\odot)$")
    ax4.set_ylabel(r"$\log\,M_{\rm BH}\ {\rm (H\beta)}\ (M_\odot)$")
    ax4.set_xlim(8.5, 10.0)
    ax4.set_ylim(8.5, 10.0)

    xmin, xmax = ax4.get_xlim()
    line = np.linspace(xmin, xmax, 100)
    ax4.plot(line, line, linestyle=":", color="black", linewidth=1.2, label="1:1 line")
    ax4.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(fig_main_bhm, f"logMBH_Hb_vs_logMBH_CIV_all_corrections_{OUTPUT_TAG}.png")
    plt.close(fig_main_bhm)


def plot_log_mbh_fitted_corrections(df):
    plot_name = "log M_BH fitted corrections"
    required = [
        "logMBH_CIV", "logMBH_CIV_err", "logMBH_CIV_fitted_corr_blueshift",
        "logMBH_CIV_fitted_corr_asym", "logMBH_Hb", "logMBH_Hb_err"
    ]
    if not has_columns(df, required, plot_name):
        return

    dfit = df[
        np.isfinite(df["logMBH_CIV"]) &
        np.isfinite(df["logMBH_CIV_err"]) &
        np.isfinite(df["logMBH_CIV_fitted_corr_blueshift"]) &
        np.isfinite(df["logMBH_CIV_fitted_corr_asym"]) &
        np.isfinite(df["logMBH_Hb"]) &
        (df["logMBH_Hb_err"] > 0) &
        np.isfinite(df["logMBH_Hb_err"])
    ].copy()
    if dfit.empty:
        print(f"Skipping {plot_name}: no rows with all required finite values")
        return

    x = dfit["logMBH_CIV"].to_numpy()
    x_bs = dfit["logMBH_CIV_fitted_corr_blueshift"].to_numpy()
    x_as = dfit["logMBH_CIV_fitted_corr_asym"].to_numpy()
    y = dfit["logMBH_Hb"].to_numpy()
    xerr = dfit["logMBH_CIV_err"].to_numpy()
    yerr = dfit["logMBH_Hb_err"].to_numpy()

    med_raw, scat_raw = median_and_inner50_scatter(x, y)
    med_bs, scat_bs = median_and_inner50_scatter(x_bs, y)
    med_as, scat_as = median_and_inner50_scatter(x_as, y)

    fig_fit = plt.figure(figsize=(6, 5))
    ax_fit = fig_fit.add_subplot(111)
    ax_fit.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="o", ms=7,
        mfc="black", mec="black", ecolor="black", elinewidth=1.2, capsize=3,
        linestyle="none",
        label=fr"Uncorrected  ($\Delta={med_raw:+.2f}$, $\sigma={scat_raw:.2f}$)"
    )
    ax_fit.errorbar(
        x_bs, y, xerr=xerr, yerr=yerr, fmt="D", ms=7,
        mfc="none", mec="red", ecolor="red", elinewidth=1.2, capsize=3,
        linestyle="none",
        label=fr"Blueshift corr.  ($\Delta={med_bs:+.2f}$, $\sigma={scat_bs:.2f}$)"
    )
    ax_fit.errorbar(
        x_as, y, xerr=xerr, yerr=yerr, fmt="s", ms=7,
        mfc="none", mec="blue", ecolor="blue", elinewidth=1.2, capsize=3,
        linestyle="none",
        label=fr"Asymmetry corr.  ($\Delta={med_as:+.2f}$, $\sigma={scat_as:.2f}$)"
    )

    ax_fit.set_xlabel(r"$\log\,M_{\rm BH}\ {\rm (C\,IV)}\ (M_\odot)$")
    ax_fit.set_ylabel(r"$\log\,M_{\rm BH}\ {\rm (H\beta)}\ (M_\odot)$")
    ax_fit.set_xlim(8.5, 10.0)
    ax_fit.set_ylim(8.5, 10.0)

    xmin, xmax = ax_fit.get_xlim()
    line = np.linspace(xmin, xmax, 100)
    ax_fit.plot(line, line, linestyle=":", color="black", linewidth=1.2, label="1:1 line")
    ax_fit.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(fig_fit, f"logMBH_Hb_vs_logMBH_CIV_fitted_corrections_{OUTPUT_TAG}.png")
    plt.close(fig_fit)


def plot_log_fwhm_fitted_corrections(df):
    plot_name = "log FWHM fitted corrections"
    required = [
        "FWHM_CIV", "FWHM_err_mcmc", "FWHM_CIV_fitted_corr_blueshift",
        "FWHM_CIV_fitted_corr_asym", "fwhm_hb", "fwhm_hb_err", "logMBH_Hb_err"
    ]
    if not has_columns(df, required, plot_name):
        return

    dfw = df[
        finite_positive(df["FWHM_CIV"]) &
        np.isfinite(df["FWHM_err_mcmc"]) &
        finite_positive(df["FWHM_CIV_fitted_corr_blueshift"]) &
        finite_positive(df["FWHM_CIV_fitted_corr_asym"]) &
        finite_positive(df["fwhm_hb"]) &
        (df["logMBH_Hb_err"] > 0) &
        np.isfinite(df["fwhm_hb_err"])
    ].copy()
    if dfw.empty:
        print(f"Skipping {plot_name}: no rows with all required finite values")
        return

    xf = np.log10(dfw["FWHM_CIV"].to_numpy())
    xf_bs = np.log10(dfw["FWHM_CIV_fitted_corr_blueshift"].to_numpy())
    xf_as = np.log10(dfw["FWHM_CIV_fitted_corr_asym"].to_numpy())
    yf = np.log10(dfw["fwhm_hb"].to_numpy())

    xerrf = dfw["FWHM_err_mcmc"].to_numpy() / (LN10 * dfw["FWHM_CIV"].to_numpy())
    yerrf = dfw["fwhm_hb_err"].to_numpy() / (LN10 * dfw["fwhm_hb"].to_numpy())

    fig_fwhm = plt.figure(figsize=(6, 5))
    ax_fwhm = fig_fwhm.add_subplot(111)
    ax_fwhm.errorbar(
        xf, yf, xerr=xerrf, yerr=yerrf, fmt="o", ms=7,
        mfc="black", mec="black", ecolor="black", elinewidth=1.2, capsize=3,
        linestyle="none", label="Uncorrected C IV"
    )
    ax_fwhm.errorbar(
        xf_bs, yf, xerr=xerrf, yerr=yerrf, fmt="D", ms=7,
        mfc="none", mec="red", ecolor="red", elinewidth=1.2, capsize=3,
        linestyle="none", label="Blueshift corrected"
    )
    ax_fwhm.errorbar(
        xf_as, yf, xerr=xerrf, yerr=yerrf, fmt="s", ms=7,
        mfc="none", mec="blue", ecolor="blue", elinewidth=1.2, capsize=3,
        linestyle="none", label="Asymmetry corrected"
    )

    ax_fwhm.set_xlabel(r"$\log\,\mathrm{FWHM}_{\rm C\,IV}\ (\mathrm{km\ s^{-1}})$")
    ax_fwhm.set_ylabel(r"$\log\,\mathrm{FWHM}_{\rm H\beta}\ (\mathrm{km\ s^{-1}})$")
    ax_fwhm.set_xlim(3.2, 4.2)
    ax_fwhm.set_ylim(3.2, 3.8)

    xmin, xmax = ax_fwhm.get_xlim()
    line = np.linspace(xmin, xmax, 100)
    ax_fwhm.plot(line, line, linestyle=":", color="black", linewidth=1.2, label="1:1 line")
    ax_fwhm.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    save_figure(fig_fwhm, f"logFWHM_Hb_vs_logFWHM_CIV_fitted_corrections_{OUTPUT_TAG}.png")
    plt.close(fig_fwhm)


def report_missing_info(df):
    expected_for_all_plots = [
        "CIV_blueshift_kms", "FWHM_CIV", "FWHM_err_mcmc",
        "FWHM_CIV_corr_blueshift", "FWHM_CIV_corr_asym",
        "fwhm_hb", "fwhm_hb_err", "logMBH_CIV", "logMBH_CIV_err",
        "logMBH_CIV_corr_blueshift", "logMBH_CIV_corr_asym",
        "logMBH_Hb", "logMBH_Hb_err",
        "logMBH_CIV_fitted_corr_blueshift", "logMBH_CIV_fitted_corr_asym",
        "FWHM_CIV_fitted_corr_blueshift", "FWHM_CIV_fitted_corr_asym"
    ]
    missing_columns = [col for col in expected_for_all_plots if col not in df.columns]
    if missing_columns:
        print("Missing info in final target values CSV for some original plots:")
        for col in missing_columns:
            print(f"  - {col}")

    empty_counts = df.isna().sum()
    empty_counts = empty_counts[empty_counts > 0]
    if not empty_counts.empty:
        print("Columns with blank/NaN values:")
        for col, count in empty_counts.items():
            print(f"  - {col}: {count}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded: {CSV_PATH}")
    print(f"Rows: {len(df)}")

    report_missing_info(df)

    plot_blueshift_hist(df)
    plot_fwhm_vs_blueshift(df)
    plot_log_fwhm_all_corrections(df)
    plot_log_mbh_all_corrections(df)
    plot_log_mbh_fitted_corrections(df)
    plot_log_fwhm_fitted_corrections(df)


if __name__ == "__main__":
    main()
