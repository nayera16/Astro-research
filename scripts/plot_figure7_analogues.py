"""Create Zuo et al. (2020) Figure 7 analogues for the current samples.

The script uses existing fit coefficients; it does not rerun LinMix.  Vertical
error bars propagate the independent C IV and H-beta FWHM uncertainties.
Horizontal error bars use the Monte Carlo blueshift and asymmetry uncertainties
for our sample and the published blueshift uncertainties for the Zuo sample.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MY_CSV = Path("/Users/nayera/PyQSOFit/Target_lists/civ_output_final_target_values_mc.csv")
ZUO_CSV = Path("/Users/nayera/PyQSOFit/Target_lists/zuo_civ_hbeta_fwhm_ratio_comparison.csv")

OUTPUT_DIR = SCRIPT_DIR.parent / "images and graphics" / "figure7_analogues"
MY_OUTPUT = OUTPUT_DIR / "figure7_my_sample.png"
MY_OUTPUT_PDF = OUTPUT_DIR / "figure7_my_sample.pdf"
COMBINED_OUTPUT = OUTPUT_DIR / "figure7_combined_blueshift.png"
COMBINED_OUTPUT_PDF = OUTPUT_DIR / "figure7_combined_blueshift.pdf"

# y = alpha + beta x, where x = blueshift / (1000 km/s) or AS_CIV.
COATMAN_BS = (0.61, 0.36)
ZUO_BS = (0.67, 0.41)
MY_BS = (0.867691455782281, 0.4621603289579128)
MY_AS = (1.4308425045451494, -1.2158792742483733)
COMBINED_BS = (0.674446216733452, 0.48976868043307425)


def ratio_and_error(civ, civ_err, hb, hb_err):
    """Return FWHM_CIV/FWHM_Hbeta and its propagated 1-sigma error."""
    ratio = civ / hb
    error = ratio * np.sqrt((civ_err / civ) ** 2 + (hb_err / hb) ** 2)
    return ratio, error


def load_my_sample():
    df = pd.read_csv(MY_CSV)
    columns = [
        "FWHM_CIV",
        "FWHM_err_mcmc",
        "fwhm_hb",
        "fwhm_hb_err",
        "CIV_blueshift_kms",
        "CIV_blueshift_err_mc",
        "CIV_asym",
        "CIV_asym_err_mc",
    ]
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(values).all(axis=1) & (
        values[["FWHM_CIV", "fwhm_hb", "CIV_blueshift_kms"]] > 0
    ).all(axis=1)
    values = values.loc[mask].copy()
    values["ratio"], values["ratio_err"] = ratio_and_error(
        values["FWHM_CIV"].to_numpy(),
        values["FWHM_err_mcmc"].to_numpy(),
        values["fwhm_hb"].to_numpy(),
        values["fwhm_hb_err"].to_numpy(),
    )
    return values


def load_zuo_sample():
    df = pd.read_csv(ZUO_CSV)
    columns = [
        "civ_fwhm_km_s",
        "civ_fwhm_err_km_s",
        "zuo_hbeta_fwhm_km_s",
        "zuo_hbeta_fwhm_err_km_s",
        "civ_delta_v_half_km_s",
        "civ_delta_v_half_err_km_s",
    ]
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    values = values.loc[mask].copy()
    values["ratio"], values["ratio_err"] = ratio_and_error(
        values["civ_fwhm_km_s"].to_numpy(),
        values["civ_fwhm_err_km_s"].to_numpy(),
        values["zuo_hbeta_fwhm_km_s"].to_numpy(),
        values["zuo_hbeta_fwhm_err_km_s"].to_numpy(),
    )
    return values


def relation(x, coefficients):
    alpha, beta = coefficients
    return alpha + beta * x


def style_axis(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=7, width=1.2)
    ax.tick_params(which="minor", length=4, width=1.0)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_my_sample(my):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)

    ax = axes[0]
    x_bs = my["CIV_blueshift_kms"].to_numpy() / 1000.0
    xerr_bs = my["CIV_blueshift_err_mc"].to_numpy() / 1000.0
    ax.errorbar(
        x_bs,
        my["ratio"],
        xerr=xerr_bs,
        yerr=my["ratio_err"],
        fmt="o",
        ms=7,
        mfc="white",
        mec="black",
        mew=1.2,
        ecolor="0.35",
        elinewidth=1.0,
        capsize=3,
        label=fr"Our sample ($N={len(my)}$)",
        zorder=3,
    )
    bs_xmax = max(4.1, float((x_bs + xerr_bs).max()) * 1.03)
    xline = np.linspace(0, bs_xmax, 300)
    ax.plot(xline, relation(xline, MY_BS), color="black", linestyle="-.", linewidth=1.8,
            label=r"Our fit: $0.87+0.46x$")
    ax.plot(xline, relation(xline, COATMAN_BS), color="red", linestyle="--", linewidth=1.8,
            label=r"Coatman+17: $0.61+0.36x$")
    ax.set_xlim(0, xline.max())
    ax.set_xlabel(r"$\Delta V_{\rm C\,IV}/(1000\ {\rm km\ s^{-1}})$")
    ax.set_ylabel(r"${\rm FWHM}_{\rm C\,IV}/{\rm FWHM}_{\rm H\beta}$")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.text(0.97, 0.96, "Blueshift", transform=ax.transAxes, ha="right", va="top")

    ax = axes[1]
    x_as = my["CIV_asym"].to_numpy()
    xerr_as = my["CIV_asym_err_mc"].to_numpy()
    ax.errorbar(
        x_as,
        my["ratio"],
        xerr=xerr_as,
        yerr=my["ratio_err"],
        fmt="o",
        ms=7,
        mfc="white",
        mec="black",
        mew=1.2,
        ecolor="0.35",
        elinewidth=1.0,
        capsize=3,
        label=fr"Our sample ($N={len(my)}$)",
        zorder=3,
    )
    asym_min = float((x_as - xerr_as).min())
    asym_max = float((x_as + xerr_as).max())
    pad = 0.05 * (asym_max - asym_min)
    xline = np.linspace(asym_min - pad, asym_max + pad, 300)
    ax.plot(xline, relation(xline, MY_AS), color="black", linestyle="-.", linewidth=1.8,
            label=r"Our fit: $1.43-1.22\,AS_{\rm C\,IV}$")
    ax.set_xlim(xline.min(), xline.max())
    ax.set_xlabel(r"$AS_{\rm C\,IV}$")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.text(0.03, 0.96, "Asymmetry", transform=ax.transAxes, ha="left", va="top")

    ymax = max(2.5, float((my["ratio"] + my["ratio_err"]).max()) * 1.08)
    axes[0].set_ylim(0, ymax)
    for axis in axes:
        style_axis(axis)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(MY_OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(MY_OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_combined(my, zuo):
    fig, ax = plt.subplots(figsize=(7.0, 5.6))

    x_my = my["CIV_blueshift_kms"].to_numpy() / 1000.0
    xerr_my = my["CIV_blueshift_err_mc"].to_numpy() / 1000.0
    x_zuo = zuo["civ_delta_v_half_km_s"].to_numpy() / 1000.0
    xerr_zuo = zuo["civ_delta_v_half_err_km_s"].to_numpy() / 1000.0

    ax.errorbar(
        x_zuo,
        zuo["ratio"],
        xerr=xerr_zuo,
        yerr=zuo["ratio_err"],
        fmt="o",
        ms=6.5,
        mfc="white",
        mec="0.35",
        mew=1.1,
        ecolor="0.55",
        elinewidth=0.9,
        capsize=2.5,
        label=fr"Zuo+20 ($N={len(zuo)}$)",
        zorder=2,
    )
    ax.errorbar(
        x_my,
        my["ratio"],
        xerr=xerr_my,
        yerr=my["ratio_err"],
        fmt="s",
        ms=6.5,
        mfc="#2878b5",
        mec="#15527c",
        mew=0.9,
        ecolor="#2878b5",
        elinewidth=1.0,
        capsize=3,
        label=fr"Our sample ($N={len(my)}$)",
        zorder=3,
    )

    combined_xmax = max(
        4.1,
        float((x_my + xerr_my).max()) * 1.03,
        float((x_zuo + xerr_zuo).max()) * 1.03,
    )
    xline = np.linspace(0, combined_xmax, 300)
    ax.plot(xline, relation(xline, COMBINED_BS), color="black", linewidth=2.2,
            label=r"Combined fit: $0.67+0.49x$")
    ax.plot(xline, relation(xline, COATMAN_BS), color="red", linestyle="--", linewidth=1.9,
            label=r"Coatman+17: $0.61+0.36x$")
    ax.plot(xline, relation(xline, ZUO_BS), color="0.45", linestyle=":", linewidth=1.4,
            label=r"Zuo+20 fit: $0.67+0.41x$")
    ax.plot(xline, relation(xline, MY_BS), color="#2878b5", linestyle="-.", linewidth=1.4,
            label=r"Our fit: $0.87+0.46x$")

    ymax = max(
        2.5,
        float((my["ratio"] + my["ratio_err"]).max()) * 1.08,
        float((zuo["ratio"] + zuo["ratio_err"]).max()) * 1.08,
    )
    ax.set_xlim(0, xline.max())
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$\Delta V_{\rm C\,IV}/(1000\ {\rm km\ s^{-1}})$")
    ax.set_ylabel(r"${\rm FWHM}_{\rm C\,IV}/{\rm FWHM}_{\rm H\beta}$")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=1)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(COMBINED_OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(COMBINED_OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    my = load_my_sample()
    zuo = load_zuo_sample()
    plot_my_sample(my)
    plot_combined(my, zuo)
    print(f"My-sample objects: {len(my)}")
    print(f"Zuo positive-blueshift objects: {len(zuo)}")
    print(f"Saved {MY_OUTPUT}")
    print(f"Saved {MY_OUTPUT_PDF}")
    print(f"Saved {COMBINED_OUTPUT}")
    print(f"Saved {COMBINED_OUTPUT_PDF}")


if __name__ == "__main__":
    main()
