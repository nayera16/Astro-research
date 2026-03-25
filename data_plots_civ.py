import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LN10 = np.log(10.0)

# style
plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

# data
csv_path = "/Users/nayera/PyQSOFit/Target_lists/civ_output_with_derived.csv"
df = pd.read_csv(csv_path)

# Histogram of C IV blueshifts

fig5 = plt.figure(figsize=(6, 5))
ax5 = fig5.add_subplot(111)
# Remove objects missing blueshift
dfh = df[np.isfinite(df["CIV_blueshift_kms"])].copy()

# Choose binning that includes negative values
v = dfh["CIV_blueshift_kms"].to_numpy()
vmin = np.floor(v.min() / 500) * 500
vmax = np.ceil(v.max() / 500) * 500

bins = np.arange(vmin, vmax + 500, 500)  # 500 km/s bins

fig5 = plt.figure(figsize=(6, 5))
ax5 = fig5.add_subplot(111)
m = max(abs(vmin), abs(vmax))
ax5.set_xlim(-m, m)
ax5.hist(v, bins=bins, color="lightgray", edgecolor="black", linewidth=1.2)

ax5.set_xlabel(r"$\Delta V_{\rm C\,IV}$  (km s$^{-1}$)")
ax5.set_ylabel("Number of quasars")

plt.tight_layout()
fig5.savefig("figure7style_blueshift_hist_all.png", dpi=300, bbox_inches="tight")
print("Saved: figure7style_blueshift_hist_all.png")


# FWHM vs C IV blueshift

fig6 = plt.figure(figsize=(6, 5))
ax6 = fig6.add_subplot(111)

ax6.scatter(
    df["CIV_blueshift_kms"],
    df["FWHM_CIV"],
    facecolors="none",
    edgecolors="black",
    linewidth=1.2,
    s=70
)

ax6.set_xlabel(r"$\Delta V_{\rm C\,IV}$  (km s$^{-1}$)")
ax6.set_ylabel(r"FWHM$_{\rm C\,IV}$  (km s$^{-1}$)")
ax6.set_xlim(0, 5000)
ax6.set_yscale("log")

plt.tight_layout()
fig6.savefig("fwhm_vs_blueshift.png", dpi=300, bbox_inches="tight")
print("Saved: fwhm_vs_blueshift.png")

# ---------------------------------------------------------
# Figure 7-style: log FWHM_Hβ vs log FWHM_CIV
# (red diamonds = corrected log FWHM_CIV using Eq. 6)
# ---------------------------------------------------------
dff = df[
    np.isfinite(df["FWHM_CIV"]) &
    np.isfinite(df["FWHM_err_mcmc"]) &
    np.isfinite(df["FWHM_CIV_corr_blueshift"]) &
    np.isfinite(df["fwhm_hb"]) &
    np.isfinite(df["fwhm_hb_err"])
].copy()

# log values
x = np.log10(dff["FWHM_CIV"].to_numpy())
x_corr = np.log10(dff["FWHM_CIV_corr_blueshift"].to_numpy())
y = np.log10(dff["fwhm_hb"].to_numpy())

# convert linear errors to dex errors
xerr = dff["FWHM_err_mcmc"].to_numpy() / (LN10 * dff["FWHM_CIV"].to_numpy())

# For corrected x: keep same fractional CIV uncertainty (good approximation)
xerr_corr = dff["FWHM_err_mcmc"].to_numpy() / (LN10 * dff["FWHM_CIV"].to_numpy())

yerr = dff["fwhm_hb_err"].to_numpy() / (LN10 * dff["fwhm_hb"].to_numpy())

fig7 = plt.figure(figsize=(6, 5))
ax7 = fig7.add_subplot(111)

# black circles (uncorrected CIV)
ax7.errorbar(
    x, y,
    xerr=xerr, yerr=yerr,
    fmt="o", ms=7,
    mfc="black", mec="black",
    ecolor="black", elinewidth=1.2, capsize=3,
    linestyle="none"
)

# red open diamonds (corrected CIV)
ax7.errorbar(
    x_corr, y,
    xerr=xerr_corr, yerr=yerr,
    fmt="D", ms=7,
    mfc="none", mec="red",
    ecolor="red", elinewidth=1.2, capsize=3,
    linestyle="none"
)

ax7.set_xlabel(r"$\log\,\mathrm{FWHM}_{\rm C\,IV}\ (\mathrm{km\ s^{-1}})$")
ax7.set_ylabel(r"$\log\,\mathrm{FWHM}_{\rm H\beta}\ (\mathrm{km\ s^{-1}})$")

# Match Zuo-ish axis bounds (adjust if your sample pushes outside)
ax7.set_xlim(3.2, 4.0)
ax7.set_ylim(3.2, 3.8)

# 1:1 correlation line (dotted)
xmin, xmax = ax7.get_xlim()
line = np.linspace(xmin, xmax, 100)
ax7.plot(line, line, linestyle=":", color="black", linewidth=1.2)
ax7.legend(["1:1 line", "Uncorrected C IV", "Corrected C IV"], loc="upper right", fontsize=12)

plt.tight_layout()
fig7.savefig("logFWHM_Hb_vs_logFWHM_CIV_corrEq6.png", dpi=300, bbox_inches="tight")
print("Saved: logFWHM_Hb_vs_logFWHM_CIV_corrEq6.png")

# ---------------------------------------------------------
# Figure 8-style: log Hβ_Mbh vs log CIV_Mbh
# (red diamonds = corrected log CIV_Mbh using Eq. 6)
# ---------------------------------------------------------
dff = df[
    np.isfinite(df["logMBH_CIV"]) &
    np.isfinite(df["logMBH_CIV_err"]) &
    np.isfinite(df["logMBH_CIV_corr_blueshift"]) &
    np.isfinite(df["logMBH_Hb"]) &
    np.isfinite(df["logMBH_Hb_err"])
].copy()

# log values
x = (dff["logMBH_CIV"].to_numpy())
x_corr = (dff["logMBH_CIV_corr_blueshift"].to_numpy())
y = (dff["logMBH_Hb"].to_numpy())

# convert linear errors to dex errors
xerr = dff["logMBH_CIV_err"].to_numpy()
xerr_corr = dff["logMBH_CIV_err"].to_numpy()
yerr = dff["logMBH_Hb_err"].to_numpy()

fig7 = plt.figure(figsize=(6, 5))
ax7 = fig7.add_subplot(111)

# black circles (uncorrected CIV)
ax7.errorbar(
    x, y,
    xerr=xerr, yerr=yerr,
    fmt="o", ms=7,
    mfc="black", mec="black",
    ecolor="black", elinewidth=1.2, capsize=3,
    linestyle="none"
)

# red open diamonds (corrected CIV)
ax7.errorbar(
    x_corr, y,
    xerr=xerr_corr, yerr=yerr,
    fmt="D", ms=7,
    mfc="none", mec="red",
    ecolor="red", elinewidth=1.2, capsize=3,
    linestyle="none"
)

ax7.set_xlabel(r"$\log\,\mathrm{MBH}_{\rm C\,IV}\ (\mathrm{M_\odot})$")
ax7.set_ylabel(r"$\log\,\mathrm{MBH}_{\rm H\beta}\ (\mathrm{M_\odot})$")

# Match Zuo-ish axis bounds (adjust if your sample pushes outside)
ax7.set_xlim(8.5, 10.0)
ax7.set_ylim(8.5, 10.0)

# 1:1 correlation line (dotted)
xmin, xmax = ax7.get_xlim()
line = np.linspace(xmin, xmax, 100)
ax7.plot(line, line, linestyle=":", color="black", linewidth=1.2)
ax7.legend(["1:1 line", "Uncorrected C IV", "Corrected C IV"], loc="upper right", fontsize=12)

plt.tight_layout()
fig7.savefig("logMBH_Hb_vs_logMBH_CIV_corrEq6.png", dpi=300, bbox_inches="tight")
print("Saved: logMBH_Hb_vs_logMBH_CIV_corrEq6.png")