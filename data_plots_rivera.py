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

# ----- EW vs CIV to compare to Rivera plot ------ #

# data points from my data
d = df[np.isfinite(df["blueshift_rivera"]) & np.isfinite(df["br_EW"])].copy()

x = d["blueshift_rivera"].to_numpy()
y = d["br_EW"].to_numpy()

fig = plt.figure(figsize=(6.5, 5.2))
ax = fig.add_subplot(111)

ax.scatter(
    x, y,
    s=70,
    color="black",
    # facecolors="none",
    # edgecolors="black",
    linewidth=1.2
)

# McCaffrey and Richards 2021 best fit line from their github repo
fit_path = "/Users/nayera/Astro-research/bestfit.npy"
fit = np.load(fit_path)

x_fit = fit[:, 0]
y_fit = 10**fit[:, 1]   # convert log10(EW) back to EW in Å

ax.set_xlim(-1500, 5000)
ax.set_ylim(0, 225)

ax.plot(
    x_fit,
    y_fit,
    color="black",
    linewidth=1.5,
    label="McCaffrey+21 best fit"
)
ax.set_title("With Rivera blueshift calculation")

# Match the axis bounds from the plot you showed
ax.set_xlim(-1500, 5000)
ax.set_ylim(0, 220)

ax.set_xlabel(r"C IV Blueshift (km s$^{-1}$)")
ax.set_ylabel(r"C IV EW ($\AA$)")

plt.tight_layout()
fig.savefig("NEW_EW_vs_CIVblueshift_mySample.png", dpi=300, bbox_inches="tight")
print("Saved: NEW_EW_vs_CIVblueshift_mySample.png")