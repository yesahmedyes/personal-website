import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from style import (
    setup, clean_axes, save, cov_ellipse, point_rgba,
    INK, AXIS, HUE, POINT, EDGE, REGION, BAND,
    FONT, LW, MARK,
)

setup()

# Two-component convention: component 1 -> blue, component 2 -> coral.
HUE2 = [HUE.blue, HUE.coral]            # ellipse borders / opaque hue
EDGE2 = [EDGE.blue, EDGE.coral]
POINT2 = [POINT.blue, POINT.coral]
REGION2 = [REGION.blue, REGION.coral]
BAND_OUT = [BAND.outer_blue, BAND.outer_coral]
BAND_IN = [BAND.inner_blue, BAND.inner_coral]


def _make_cov(sx, sy, deg):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R @ np.diag([sx ** 2, sy ** 2]) @ R.T


# ---------------------------------------------------------------------------
# 1. The latent-variable story: the same data with and without its labels.
#    Left: all we ever observe is the unlabeled cloud x^(i). Right: the model's
#    explanation -- a hidden z^(i) chose one of two Gaussians for each point.
# ---------------------------------------------------------------------------
def figure_latent_structure():
    means = [np.array([-2.3, 0.9]), np.array([2.2, -0.8])]
    covs = [_make_cov(1.5, 0.62, 28), _make_cov(1.35, 0.7, -18)]

    rng = np.random.default_rng(12)
    X0 = rng.multivariate_normal(means[0], covs[0], size=65)
    X1 = rng.multivariate_normal(means[1], covs[1], size=65)
    X = np.vstack([X0, X1])
    z = np.array([0] * len(X0) + [1] * len(X1))

    pad = 1.15
    cx = 0.5 * (X[:, 0].min() + X[:, 0].max())
    cy = 0.5 * (X[:, 1].min() + X[:, 1].max())
    half = 0.5 * max(np.ptp(X[:, 0]), np.ptp(X[:, 1])) + pad
    xlim, ylim = (cx - half, cx + half), (cy - half, cy + half)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for ax in axes:
        clean_axes(ax)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Left: the training set as we actually receive it -- no labels.
    axes[0].set_title(r"observed: only $x^{(i)}$", fontsize=FONT.title,
                      color=INK, pad=8)
    axes[0].scatter(X[:, 0], X[:, 1], s=MARK.point, facecolor=POINT.muted,
                    edgecolor=INK, linewidth=LW.edge, zorder=3)

    # Right: the hidden structure -- each z picks the Gaussian its x came from.
    # Density bands styled like the GLA generative panel: solid point-colour
    # borders over layered translucent fills (2-sigma light, 1-sigma point), no
    # centroid markers; neutral labels keep the near-black anchor.
    axes[1].set_title(r"latent: $z^{(i)}$ picks the Gaussian", fontsize=FONT.title,
                      color=INK, pad=8)
    for j in range(2):
        cov_ellipse(axes[1], means[j], covs[j], 2.0, border=HUE2[j],
                    fill=BAND_OUT[j], z=1)
        cov_ellipse(axes[1], means[j], covs[j], 1.0, border=HUE2[j],
                    fill=BAND_IN[j], z=2)
        m = z == j
        axes[1].scatter(X[m, 0], X[m, 1], s=MARK.point, facecolor=POINT2[j],
                        edgecolor=EDGE2[j], linewidth=LW.edge, zorder=4)

    axes[1].text(means[0][0], means[0][1] + 2.6,
                 r"$x \mid z{=}0 \sim \mathcal{N}(\mu_0, \Sigma_0)$",
                 ha="center", va="bottom", fontsize=FONT.annotation,
                 color=INK, zorder=8)
    axes[1].text(means[1][0], means[1][1] - 2.6,
                 r"$x \mid z{=}1 \sim \mathcal{N}(\mu_1, \Sigma_1)$",
                 ha="center", va="top", fontsize=FONT.annotation,
                 color=INK, zorder=8)

    fig.subplots_adjust(wspace=0.16)
    save(fig, "gmm-latent-structure.png")


# ---------------------------------------------------------------------------
# 2. The E-step in one dimension. Top: the two weighted component densities and
#    the mixture they sum to. Bottom: the responsibility w_1(x) = p(z=1 | x),
#    which moves smoothly from 1 to 0 -- a soft assignment, not a hard one.
# ---------------------------------------------------------------------------
def figure_responsibilities():
    mu, sd, phi = [-1.9, 1.9], [1.15, 1.3], [0.5, 0.5]

    def gauss(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    xs = np.linspace(-6.2, 6.4, 1200)
    comp = [phi[j] * gauss(xs, mu[j], sd[j]) for j in range(2)]
    total = comp[0] + comp[1]
    w1 = comp[0] / total
    x_star = xs[int(np.argmin(np.abs(w1 - 0.5)))]

    # A handful of samples from the mixture, placed on the responsibility curve.
    rng = np.random.default_rng(5)
    zs = rng.integers(0, 2, size=46)
    pts = rng.normal(np.array(mu)[zs], np.array(sd)[zs])
    pts = pts[(pts > xs[0] + 0.2) & (pts < xs[-1] - 0.2)]
    w_pts = (phi[0] * gauss(pts, mu[0], sd[0])) / (
        phi[0] * gauss(pts, mu[0], sd[0]) + phi[1] * gauss(pts, mu[1], sd[1]))

    c_blue, c_coral = np.array(to_rgb(HUE.blue)), np.array(to_rgb(HUE.coral))
    e_blue, e_coral = np.array(to_rgb(EDGE.blue)), np.array(to_rgb(EDGE.coral))
    face = w_pts[:, None] * c_blue + (1 - w_pts)[:, None] * c_coral
    edge = w_pts[:, None] * e_blue + (1 - w_pts)[:, None] * e_coral

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 5.9), sharex=True,
        gridspec_kw=dict(height_ratios=[1.5, 1.0], hspace=0.18))
    for ax in (ax1, ax2):
        clean_axes(ax)
        ax.set_xlim(xs[0], xs[-1])

    # Top: the two weighted components and the mixture density they sum to.
    for j in range(2):
        ax1.fill_between(xs, comp[j], color=REGION2[j], zorder=1)
        ax1.plot(xs, comp[j], color=EDGE2[j], lw=LW.line, zorder=3)
    ax1.plot(xs, total, color=INK, lw=LW.line, zorder=4)
    ax1.set_ylim(0, total.max() * 1.38)

    peak = [phi[j] * gauss(mu[j], mu[j], sd[j]) for j in range(2)]
    ax1.text(mu[0] - 2.1, peak[0] - 0.01, r"$\phi_0 \, p(x \mid z{=}0)$",
             ha="center", va="bottom", fontsize=FONT.annotation, color=EDGE.blue)
    ax1.text(mu[1] + 2.1, peak[1] - 0.01, r"$\phi_1 \, p(x \mid z{=}1)$",
             ha="center", va="bottom", fontsize=FONT.annotation, color=EDGE.coral)
    mid = (xs > mu[0]) & (xs < mu[1])
    dip_i = int(np.argmin(np.where(mid, total, np.inf)))
    ax1.text(xs[dip_i], total[dip_i] + 0.02, r"$p(x)$", ha="center", va="bottom",
             fontsize=FONT.annotation, color=INK)

    # Bottom: the posterior responsibility, with sampled points sitting on it.
    for y in (0.0, 0.5, 1.0):
        ax2.axhline(y, color=AXIS, lw=LW.guide,
                    linestyle=(0, (4, 3)), zorder=1)
        ax2.text(xs[0] - 0.25, y, f"{y:g}", ha="right", va="center",
                 fontsize=FONT.tick, color=INK)
    ax2.plot(xs, w1, color=EDGE.blue, lw=LW.line, zorder=3)
    ax2.plot(xs, 1 - w1, color=EDGE.coral, lw=LW.line, zorder=3)
    ax2.scatter(pts, w_pts, s=MARK.small, facecolor=point_rgba(face),
                edgecolor=edge, linewidth=LW.edge, zorder=4)
    ax2.scatter(pts, 1 - w_pts, s=MARK.small, facecolor=point_rgba(face),
                edgecolor=edge, linewidth=LW.edge, zorder=4)
    ax2.set_ylim(-0.1, 1.18)
    ax2.text(xs[0] + 0.35, 1.06, r"$w_0^{(i)} = p(z{=}0 \mid x)$", ha="left",
             va="bottom", fontsize=FONT.annotation, color=EDGE.blue)
    ax2.text(xs[-1] - 0.35, 1.06, r"$w_1^{(i)} = p(z{=}1 \mid x)$", ha="right",
             va="bottom", fontsize=FONT.annotation, color=EDGE.coral)

    # The crossover where both responsibilities equal 1/2.
    ax2.plot([x_star, x_star], [-0.1, 1.0], color=INK, lw=LW.guide,
             linestyle=(0, (4, 3)), zorder=2)

    save(fig, "gmm-responsibilities.png")


def main():
    figure_latent_structure()
    figure_responsibilities()


if __name__ == "__main__":
    main()
