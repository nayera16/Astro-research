import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# import richardsplot as rplot
# from sklearn.preprocessing import scale

from scripts.CIVfunctions_parallel import project, CIV_distance

csv_path = "/Users/nayera/PyQSOFit/Target_lists/civ_output_with_derived.csv"
dat = pd.read_csv(csv_path)
dat.iloc[:,10:].head()

ew   = dat["br_EW"].values
logEW= np.log10(ew)
blue = dat["blueshift_rivera"].values

bestfit = np.load("data/bestfit.npy")


fig = plt.figure(figsize=(6.5, 5.2))
ax = fig.add_subplot(111)

#Visualize increasing CIV distance
# style = "Simple, tail_width=1.25, head_width=20, head_length=20"
# kw = dict(arrowstyle=style, color="k")
# a1 = patches.FancyArrowPatch((2000, 2.25), (4900, 1.7),
#                              connectionstyle="arc3,rad=.47", **kw)
# plt.gca().add_patch(a1)
# plt.text(x=2800,y=2.1,s="Increasing", fontsize=30, weight='extra bold')
# plt.text(x=2735,y=1.985,s="C$_{IV}$ Distance", fontsize=30)

ax.plot(bestfit[:,0], bestfit[:,1])
ax.scatter(blue, logEW)
ax.set_xlabel("CIV Blueshift (km s$^{-1}$)", fontsize=20)
ax.set_ylabel("log$_{10}$ CIV EW (Å)", fontsize=20)
ax.set_xlim(-850,3800)
ax.set_ylim(0.9,2.4)

# plt.show()

fig.savefig("2_EW_vs_CIVblueshift_mySample.png", dpi=300, bbox_inches="tight")
print("Saved: 2_EW_vs_CIVblueshift_mySample.png")

## calculating distance
fit = np.load("data/bestfit.npy")
data = np.column_stack([blue, logEW])   # shape (N, 2): [blueshift, log10(EW)]
civ_distances = CIV_distance(data, fit)

print(civ_distances)

# print(dat.loc[civ_distances < 1e-4])

# -----------------------------------------------
# ------ CIV parallel distance vs Hb BHM --------
# -----------------------------------------------

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
csv_path = "/Users/nayera/PyQSOFit/Target_lists/civ_output_with_parallel_distance.csv"
df = pd.read_csv(csv_path)

# ----- EW vs CIV to compare to Rivera plot ------ #

# data points from my data
d = df[np.isfinite(df["CIV_parallel_distance_scaled"]) & np.isfinite(df["logMBH_Hb"])].copy()

x = d["logMBH_Hb"].to_numpy()
y = d["CIV_parallel_distance_scaled"].to_numpy()

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

ax.set_title("CIV distance vs hb mass")

ax.set_xlabel(r"logMBH_Hb")
ax.set_ylabel(r"C IV // distance")

plt.tight_layout()
fig.savefig("civ_parallel_vs_hbBHM.png", dpi=300, bbox_inches="tight")
print("Saved: civ_parallel_vs_hbBHM.png")
