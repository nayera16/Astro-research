"""Fit the uncorrected C IV--H-beta FWHM relation with LINMIX.

The fitted model is

    log10(FWHM_CIV) = alpha + beta * log10(FWHM_Hbeta) + epsilon,

where epsilon is the intrinsic scatter modeled by LINMIX. Measurement
uncertainties in both FWHM values are propagated into log10 space.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import linmix
except ModuleNotFoundError:
    # Support the local linmix checkout at <repository>/linmix/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linmix"))
    import linmix


DEFAULT_INPUT = Path(
    "/Users/nayera/PyQSOFit/Target_lists/civ_output_final_target_values.csv"
)
REQUIRED_COLUMNS = (
    "FWHM_CIV",
    "FWHM_err_mcmc",
    "fwhm_hb",
    "fwhm_hb_err",
)


def credible_summary(samples: np.ndarray) -> tuple[float, float, float]:
    """Return the posterior median and equal-tailed 95% credible interval."""
    median = float(np.median(samples))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return median, float(lower), float(upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit uncorrected log FWHM(C IV) versus log FWHM(H-beta)."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input catalog (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to LINMIX (default: 42)",
    )
    parser.add_argument(
        "--posterior-output",
        type=Path,
        help="Optional CSV path for the alpha, beta, scatter, and correlation draws",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    values = df.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    fit_mask = np.isfinite(values).all(axis=1)
    fit_mask &= (values.loc[:, REQUIRED_COLUMNS] > 0).all(axis=1)
    fitted = values.loc[fit_mask]

    if len(fitted) < 3:
        raise ValueError(f"Only {len(fitted)} valid objects; at least 3 are required")

    fwhm_hb = fitted["fwhm_hb"].to_numpy()
    fwhm_civ = fitted["FWHM_CIV"].to_numpy()
    x = np.log10(fwhm_hb)
    y = np.log10(fwhm_civ)

    # First-order propagation: sigma_log10(F) = sigma_F / (F ln(10)).
    xsig = fitted["fwhm_hb_err"].to_numpy() / (fwhm_hb * np.log(10.0))
    ysig = fitted["FWHM_err_mcmc"].to_numpy() / (fwhm_civ * np.log(10.0))

    model = linmix.LinMix(x, y, xsig=xsig, ysig=ysig, seed=args.seed)
    model.run_mcmc(silent=True)

    alpha_samples = model.chain["alpha"]
    beta_samples = model.chain["beta"]
    scatter_samples = np.sqrt(model.chain["sigsqr"])
    corr_samples = model.chain["corr"]

    alpha = credible_summary(alpha_samples)
    beta = credible_summary(beta_samples)
    scatter = credible_summary(scatter_samples)
    corr = credible_summary(corr_samples)
    probability_positive = float(np.mean(beta_samples > 0.0))
    probability_nonpositive = 1.0 - probability_positive

    print(f"Objects in catalog: {len(df)}")
    print(f"Objects used in fit: {len(fitted)}")
    print(f"Objects excluded:    {len(df) - len(fitted)}")
    print()
    print("Model:")
    print("  log10(FWHM_CIV) = alpha + beta log10(FWHM_Hbeta) + epsilon")
    print()
    print("Posterior summaries (median [95% credible interval]):")
    print(f"  alpha             = {alpha[0]:.4f} [{alpha[1]:.4f}, {alpha[2]:.4f}]")
    print(f"  beta              = {beta[0]:.4f} [{beta[1]:.4f}, {beta[2]:.4f}]")
    print(
        f"  intrinsic scatter = {scatter[0]:.4f} "
        f"[{scatter[1]:.4f}, {scatter[2]:.4f}] dex"
    )
    print(f"  latent corr. rho  = {corr[0]:.4f} [{corr[1]:.4f}, {corr[2]:.4f}]")
    print()
    print(f"P(beta > 0 | data)  = {probability_positive:.6f}")
    print(f"P(beta <= 0 | data) = {probability_nonpositive:.6f}")

    if args.posterior_output is not None:
        args.posterior_output.parent.mkdir(parents=True, exist_ok=True)
        posterior = pd.DataFrame(
            {
                "alpha": alpha_samples,
                "beta": beta_samples,
                "intrinsic_scatter_dex": scatter_samples,
                "corr": corr_samples,
            }
        )
        posterior.to_csv(args.posterior_output, index=False)
        print(f"Posterior draws saved to: {args.posterior_output}")


if __name__ == "__main__":
    main()
