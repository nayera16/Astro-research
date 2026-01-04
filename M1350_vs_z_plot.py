import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

base_path = "/Users/nayera/PyQSOFit/"
input_csv = base_path + "Target_lists/target_list_info_for_calc.csv"

targets_root = base_path + "CIV_galaxy_spectra_targets/"

# Rest wavelength
LAMBDA_A = 1350.0  # Angstroms

# AB magnitude zero-point constant for Lnu in erg/s/Hz:
# M_AB = -2.5 log10(Lnu) + 51.60
AB_CONST = 51.60

def logL1350_from_logL1450_and_slope(logL1450, alpha):
    """
    PyQSOFit L1450 is typically log10(lambda L_lambda) in erg/s at 1450 Å.
    If F_lambda ∝ lambda^alpha, then lambda L_lambda ∝ lambda^(alpha+1):

    logL1350 = logL1450 + (alpha + 1) * log10(1350/1450)
    """
    return logL1450 + (alpha + 1.0) * np.log10(1350.0 / 1450.0)


def M_AB_from_log_lambdaLlambda(log_lambdaLlambda, lambda_A=LAMBDA_A):
    """
    Convert log10(lambda L_lambda) [erg/s] at wavelength lambda into absolute AB magnitude.

    Steps:
      lambda L_lambda = lambda * L_lambda
      L_nu = (lambda / c) * (lambda L_lambda)

    using lambda in cm and c in cm/s.

    Then:
      M_AB = -2.5 log10(L_nu) + 51.60
    """
    c_cms = 2.99792458e10  # cm/s
    lambda_cm = lambda_A * 1e-8  # Å -> cm

    # log10(Lnu) = log10(lambdaLlambda) + log10(lambda/c)
    log_Lnu = log_lambdaLlambda + np.log10(lambda_cm / c_cms)
    M_AB = -2.5 * log_Lnu + AB_CONST
    return M_AB


# -----------------------------
# READ FITS + BUILD TABLE
# -----------------------------

def read_one_target(target, fits_stub):
    fits_path = f"{targets_root}{target}/output/{fits_stub}.fits"
    with fits.open(fits_path) as f:
        row = f[1].data[0]
        z = float(row["redshift"])
        logL1450 = float(row["L1450"])
        alpha = float(row["PL_slope"])

    # Guard against flagged continuum
    if not np.isfinite(logL1450) or logL1450 < 40:
        return None

    logL1350 = logL1350_from_logL1450_and_slope(logL1450, alpha)
    M1350 = M_AB_from_log_lambdaLlambda(logL1350, lambda_A=LAMBDA_A)
    return z, M1350, logL1350


def get_break_limits(zvals):
    """
    If there's a big gap in sorted redshifts, create a broken x-axis like the example.
    Returns None if no obvious break.
    """
    z = np.sort(np.array(zvals))
    if len(z) < 6:
        return None

    gaps = z[1:] - z[:-1]
    imax = np.argmax(gaps)
    maxgap = gaps[imax]

    # If there's a large empty region, break it
    if maxgap > 0.4:
        left_max = z[imax] + 0.02
        right_min = z[imax + 1] - 0.02
        return (z.min() - 0.02, left_max), (right_min, z.max() + 0.02)

    return None


# -----------------------------
# PLOTTING (Zuo-like style)
# -----------------------------

def plot_single_axis(z, M):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.scatter(z, M, marker="*", s=280, color="#e53935", edgecolor="#e53935", label="This work", zorder=3)

    ax.set_xlabel("Redshift", fontsize=32)
    ax.set_ylabel(r"$M_{1350}$", fontsize=38)

    # Make it look like the example (magnitudes: more negative higher up)
    # If you want inverted, uncomment next line:
    # ax.invert_yaxis()

    ax.legend(frameon=True, fontsize=20, loc="lower right")
    fig.tight_layout()
    fig.savefig("M1350_vs_redshift.png", dpi=300)
    print("Saved: M1350_vs_redshift.png")


def plot_broken_axis(z, M, left_xlim, right_xlim):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05}
    )

    for ax in (ax1, ax2):
        ax.scatter(z, M, marker="*", s=280, color="#e53935", edgecolor="#e53935", zorder=3)

    ax1.set_xlim(*left_xlim)
    ax2.set_xlim(*right_xlim)

    ax1.set_xlabel("Redshift", fontsize=32)
    ax2.set_xlabel("Redshift", fontsize=32)
    ax1.set_ylabel(r"$M_{1350}$", fontsize=38)

    # Hide spines between plots
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(labelleft=False)

    # Diagonal break marks
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=2.0)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    # One legend (put it on right axis like the example)
    ax2.legend(["This work"], frameon=True, fontsize=20, loc="lower right")

    fig.tight_layout()
    fig.savefig("M1350_vs_redshift_broken.png", dpi=300)
    print("Saved: M1350_vs_redshift_broken.png")


def main():
    df = pd.read_csv(input_csv)

    z_list, M_list, logL1350_list, good_targets = [], [], [], []

    for _, row in df.iterrows():
        target = row["Target"]
        fits_stub = row["fits_path"]  # without ".fits" (your CSV already stores stub)
        out = read_one_target(target, fits_stub)
        if out is None:
            print(f"[SKIP] {target}: bad/flagged continuum (logL1450 invalid)")
            continue
        z, M1350, logL1350 = out
        z_list.append(z)
        M_list.append(M1350)
        logL1350_list.append(logL1350)
        good_targets.append(target)

    if len(z_list) == 0:
        raise RuntimeError("No valid targets found. Check FITS paths / continuum flags.")

    # Auto decide whether to use broken axis
    br = get_break_limits(z_list)
    if br is None:
        plot_single_axis(z_list, M_list)
    else:
        left_xlim, right_xlim = br
        plot_broken_axis(z_list, M_list, left_xlim, right_xlim)

    # Optional: write a small CSV with computed values (handy for Excel)
    out_df = pd.DataFrame({
        "Target": good_targets,
        "redshift": z_list,
        "logL1350": logL1350_list,
        "M1350_AB": M_list
    })
    out_df.to_csv("M1350_values.csv", index=False)
    print("Saved: M1350_values.csv")


if __name__ == "__main__":
    main()
