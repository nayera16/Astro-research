import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
bins = np.arange(0, 5500, 1000)  # 0–7000 in increments of 500
ax5.hist(
    df["CIV_blueshift_kms"], 
    bins=bins, 
    edgecolor="black", 
    #linewidth=1.2
)

ax5.set_xlim(0, 5000)
ax5.set_xlabel(r"$\Delta V_{\rm C\,IV}$  (km s$^{-1}$)")
ax5.set_ylabel("Number of quasars")

#plt.tight_layout()
fig5.savefig("blueshift_hist.png", dpi=300, bbox_inches="tight")
print("Saved: blueshift_hist.png")


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
