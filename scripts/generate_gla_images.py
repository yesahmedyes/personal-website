import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (
    setup, clean_axes, save, cov_ellipse,
    INK, HUE, POINT, EDGE, BAND, REGION,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# Two classes with distinct means and a SHARED covariance, so the GDA boundary
# is linear; this data and boundary back the generative/discriminative figure.
# ---------------------------------------------------------------------------
MU0 = np.array([-1.7, -1.0])
MU1 = np.array([1.8, 1.15])
SIGMA = np.array([[1.5, 0.6], [0.6, 0.95]])
XLIM = (-5.4, 5.8)
YLIM = (-4.4, 4.8)


def _sample():
    rng = np.random.default_rng(2)
    X0 = rng.multivariate_normal(MU0, SIGMA, 45)
    X1 = rng.multivariate_normal(MU1, SIGMA, 45)
    return X0, X1


def _boundary(phi=0.5):
    """w, b for the GDA log-posterior-ratio boundary w . x + b = 0."""
    Si = np.linalg.inv(SIGMA)
    w = Si @ (MU1 - MU0)
    b = -0.5 * (MU1 @ Si @ MU1 - MU0 @ Si @ MU0) + np.log(phi / (1 - phi))
    return w, b


def _scatter(ax, X0, X1):
    # Class convention matches classification-decision-boundary.png: y=0 blue, y=1 coral.
    ax.scatter(X0[:, 0], X0[:, 1], s=MARK.point, facecolor=POINT.blue,
               edgecolor=EDGE.blue, linewidth=LW.edge, zorder=4)
    ax.scatter(X1[:, 0], X1[:, 1], s=MARK.point, facecolor=POINT.coral,
               edgecolor=EDGE.coral, linewidth=LW.edge, zorder=4)


# ---------------------------------------------------------------------------
# Generative vs. discriminative: model the boundary, or model each class
# ---------------------------------------------------------------------------
def figure_gen_vs_disc():
    """Discriminative learning carves the input space with a boundary p(y | x);
    generative learning instead models each class's density p(x | y)."""
    X0, X1 = _sample()
    w, b = _boundary()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.4))
    for ax in (axL, axR):
        clean_axes(ax)
        ax.set_aspect("equal")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)

    def class_labels(ax):
        ax.text(-3.6, 0.5, r"$y = 0$", color=INK, fontsize=FONT.emphasis, ha="center")
        ax.text(4.3, 3.5, r"$y = 1$", color=INK, fontsize=FONT.emphasis, ha="center")

    # ----- left: discriminative -----
    gx = np.linspace(*XLIM, 200)
    gy = np.linspace(*YLIM, 200)
    XX, YY = np.meshgrid(gx, gy)
    Z = w[0] * XX + w[1] * YY + b
    axL.contourf(XX, YY, Z, levels=[-1e9, 0, 1e9],
                 colors=[REGION.blue, REGION.coral], zorder=0)
    _scatter(axL, X0, X1)
    xs = np.array(XLIM)
    axL.plot(xs, -(w[0] * xs + b) / w[1], color=INK, linewidth=LW.line, zorder=5)
    class_labels(axL)
    axL.set_title(r"Discriminative: learn $p(y \mid x)$", fontsize=FONT.title,
                  color=INK, pad=8)

    # ----- right: generative (filled density bands, black borders) -----
    for mu, hue, band_outer, band_inner in (
            (MU0, HUE.blue, BAND.outer_blue, BAND.inner_blue),
            (MU1, HUE.coral, BAND.outer_coral, BAND.inner_coral)):
        cov_ellipse(axR, mu, SIGMA, 2.0, border=hue, fill=band_outer, z=1)
        cov_ellipse(axR, mu, SIGMA, 1.0, border=hue, fill=band_inner, z=2)
    _scatter(axR, X0, X1)
    class_labels(axR)
    axR.set_title(r"Generative: model $p(x \mid y)\, p(y)$", fontsize=FONT.title,
                  color=INK, pad=8)

    fig.subplots_adjust(wspace=0.1)
    save(fig, "gla-generative-vs-discriminative.png")


def main():
    figure_gen_vs_disc()


if __name__ == "__main__":
    main()
