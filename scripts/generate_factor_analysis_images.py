import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (
    setup, clean_axes, save, cov_ellipse,
    INK, AXIS, HUE, POINT, EDGE, BAND,
    FONT, LW, MARK,
)

setup()


# Shared 1-factor model in d = 2.
MU = np.array([0.4, 0.2])
W = np.array([1.25, 0.75])               # column space of W: a line through mu
PSI = np.diag([0.55 ** 2, 0.5 ** 2])     # diagonal per-coordinate noise

WN = W / np.linalg.norm(W)               # unit vector along the line


def _data():
    rng = np.random.default_rng(11)
    z = rng.normal(0, 1, 85)
    return MU[None, :] + z[:, None] * W[None, :] \
        + rng.normal(0, [0.55, 0.5], size=(len(z), 2))


# ---------------------------------------------------------------------------
# 1. Why diagonal Sigma is not enough, in one frame: the axis-aligned fit
#    (dashed) against the tilted WW^T + Psi bands, on the same data.
# ---------------------------------------------------------------------------
def figure_covariance():
    X = _data()
    m = X.mean(axis=0)
    cov_diag = np.diag(X.var(axis=0))
    cov_fa = np.outer(W, W) + PSI

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    clean_axes(ax)
    ax.set_aspect("equal")
    pad = 1.15
    cx = 0.5 * (X[:, 0].min() + X[:, 0].max())
    cy = 0.5 * (X[:, 1].min() + X[:, 1].max())
    half = 0.5 * max(np.ptp(X[:, 0]), np.ptp(X[:, 1])) + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    cov_ellipse(ax, m, cov_fa, 2.0, border=HUE.blue, fill=BAND.outer_blue, z=1)
    cov_ellipse(ax, m, cov_fa, 1.0, border=HUE.blue, fill=BAND.inner_blue, z=2)
    cov_ellipse(ax, m, cov_diag, 2.0, border=AXIS, fill="none", ls="dashed", z=3)
    ax.scatter(X[:, 0], X[:, 1], s=MARK.point, facecolor=POINT.blue,
               edgecolor=EDGE.blue, linewidth=LW.edge, zorder=4)

    sd = 2.0 * np.sqrt(np.diag(cov_diag))
    ax.text(m[0] - 1.1, m[1] + sd[1] + 0.32, r"diagonal $\Sigma$",
            ha="center", va="bottom", fontsize=FONT.annotation,
            color=INK, zorder=8)
    lam_max = np.linalg.eigvalsh(cov_fa)[-1]
    tip = m + 2.0 * np.sqrt(lam_max) * WN
    ax.text(tip[0] + 0.15, tip[1] + 0.7, r"$WW^T + \Psi$", ha="center",
            va="bottom", fontsize=FONT.annotation, color=EDGE.blue, zorder=8)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    save(fig, "factor-analysis-covariance.png")


def main():
    figure_covariance()


if __name__ == "__main__":
    main()
