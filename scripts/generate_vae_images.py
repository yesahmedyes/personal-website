import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

from style import (
    setup, save,
    INK, AXIS, BACKGROUND,
    HUE, EDGE, TINT,
    FONT, LW,
)

setup()


# --- shared schematic helpers ---------------------------------------------
def box(ax, cx, cy, w, h, label, face, edge, fontsize=FONT.emphasis, lw=LW.line):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=face,
                           edgecolor=edge, linewidth=lw, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=4)


def disc(ax, cx, cy, r, label, face, edge, fontsize=FONT.emphasis, lw=LW.line):
    ax.add_patch(Circle((cx, cy), r, facecolor=face, edgecolor=edge,
                        linewidth=lw, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=4)


def arrow(ax, p, q, color=INK, lw=LW.line, rad=0.0, ls="solid"):
    ax.annotate("", xy=q, xytext=p, zorder=2, arrowprops=dict(
        arrowstyle="-|>", color=color, linewidth=lw, mutation_scale=15,
        shrinkA=2, shrinkB=2, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------------------
# 1. The VAE architecture: encoder heads -> latent -> decoder
# ---------------------------------------------------------------------------
def figure_architecture():
    """x is encoded into the parameters of Q (mean q(x;phi), variance v(x;psi));
    we sample a latent z from Q and decode it with g(z;theta) to reconstruct x."""
    fig, ax = plt.subplots(figsize=(11.5, 3.45))
    ax.set_axis_off()
    ax.set_xlim(0, 14)
    ax.set_ylim(1.4, 5.6)
    CY = 3.5

    # Input / output data nodes.
    box(ax, 1.0, CY, 0.9, 0.9, "$x$", BACKGROUND, AXIS)
    box(ax, 12.9, CY, 0.9, 0.9, r"$\hat{x}$", BACKGROUND,
        AXIS)

    # Encoder funnel (narrows to the right) and decoder funnel (widens).
    ax.add_patch(Polygon([(2.3, 2.0), (2.3, 5.0), (4.5, 4.3), (4.5, 2.7)],
                         closed=True, facecolor=TINT.blue,
                         edgecolor=EDGE.blue, linewidth=LW.line, zorder=3))
    ax.text(3.4, CY, "encoder", ha="center", va="center", fontsize=FONT.title,
            color=INK, zorder=4)

    ax.add_patch(Polygon([(9.4, 2.7), (9.4, 4.3), (11.6, 5.0), (11.6, 2.0)],
                         closed=True, facecolor=TINT.coral,
                         edgecolor=EDGE.coral, linewidth=LW.line, zorder=3))
    ax.text(10.55, 3.78, "decoder", ha="center", va="center", fontsize=FONT.title,
            color=INK, zorder=4)
    ax.text(10.55, 3.18, r"$g(z;\theta)$", ha="center", va="center", fontsize=FONT.annotation,
            color=INK, zorder=4)

    # Mean and variance heads.
    box(ax, 5.9, 4.45, 1.5, 0.9, r"$q(x;\phi)$", BACKGROUND,
        EDGE.blue, fontsize=FONT.title)
    box(ax, 5.9, 2.55, 1.5, 0.9, r"$v(x;\psi)$", BACKGROUND,
        EDGE.blue, fontsize=FONT.title)
    ax.text(5.9, 5.18, r"mean $\mu$", ha="center", va="center", fontsize=FONT.tick,
            color=EDGE.blue)
    ax.text(5.9, 1.82, r"variance $\sigma^2$", ha="center", va="center",
            fontsize=FONT.tick, color=EDGE.blue)

    # Latent node.
    disc(ax, 8.3, CY, 0.55, "$z$", TINT.sage, EDGE.green)
    ax.text(8.3, 4.62, r"sample $z \sim Q$", ha="center", va="center",
            fontsize=FONT.tick, color=EDGE.green)

    # Wiring.
    arrow(ax, (1.45, CY), (2.28, CY))
    arrow(ax, (4.5, CY), (5.12, 4.45))
    arrow(ax, (4.5, CY), (5.12, 2.55))
    arrow(ax, (6.68, 4.45), (7.86, 3.78))
    arrow(ax, (6.68, 2.55), (7.86, 3.22))
    arrow(ax, (8.86, CY), (9.42, CY))
    arrow(ax, (11.6, CY), (12.42, CY))

    save(fig, "vae-architecture.png")


# ---------------------------------------------------------------------------
# 2. The reparameterization trick: moving randomness off the gradient path
# ---------------------------------------------------------------------------
def figure_reparameterization():
    """Drawing z ~ N(mu, sigma^2) is the same as drawing fixed noise
    xi ~ N(0, 1) and applying the deterministic, differentiable map
    z = mu + sigma * xi. The randomness lives entirely in xi; mu and sigma
    only shift and scale it, so the gradient passes straight through them."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_axis_off()
    ax.set_xlim(0, 15.4)
    ax.set_ylim(1.1, 6.4)

    y0 = 2.4
    t = np.linspace(-3, 3, 240)
    bell = np.exp(-t ** 2 / 2)

    # Left: the fixed standard-normal noise source.
    cxL, wL, hL = 3.0, 0.82, 2.55
    ax.fill_between(cxL + wL * t, y0, y0 + hL * bell, color=TINT.coral,
                    edgecolor=EDGE.coral, linewidth=LW.line, zorder=3)
    # Right: the latent distribution, the same bell shifted by mu and widened by sigma.
    cxR, wR, hR = 11.7, 1.32, 1.62
    ax.fill_between(cxR + wR * t, y0, y0 + hR * bell, color=TINT.blue,
                    edgecolor=EDGE.blue, linewidth=LW.line, zorder=3)

    # The same handful of draws, carried through the map: a tight cluster on the
    # left fans out to a wide, shifted one on the right -- shift + scale, visibly.
    for s in (-1.7, -0.7, 0.2, 1.1, 2.1):
        xl, xr = cxL + wL * s, cxR + wR * s
        ax.annotate("", xy=(xr, y0), xytext=(xl, y0), zorder=4,
                    arrowprops=dict(arrowstyle="-", color=AXIS,
                                    linewidth=LW.guide,
                                    connectionstyle="arc3,rad=-0.32"))
        ax.plot(xl, y0, "o", ms=6, color=EDGE.coral, zorder=5)  # noqa: style -- plot marker diameter, not the s= area scale
        ax.plot(xr, y0, "o", ms=6, color=EDGE.blue, zorder=5)  # noqa: style -- plot marker diameter, not the s= area scale

    # The deterministic map, centered between the two distributions and lifted
    # clear of the fanning connectors below it.
    cxM = (cxL + cxR) / 2
    arrow(ax, (cxM - 0.95, 5.3), (cxM + 0.95, 5.3), color=HUE.green,
          lw=LW.arrow)
    ax.text(cxM, 5.95, r"$z = \mu + \sigma \odot \xi$", ha="center",
            va="center", fontsize=FONT.title, color=EDGE.green)
    ax.text(cxM, 4.78, "differentiable", ha="center", va="center",
            fontsize=FONT.tick, color=AXIS)

    # Per-distribution captions.
    ax.text(cxL, 1.6, r"$\xi \sim \mathcal{N}(0,\,1)$", ha="center",
            fontsize=FONT.title, color=EDGE.coral)
    ax.text(cxR, 1.6, r"$z \sim \mathcal{N}(\mu,\,\sigma^2)$", ha="center",
            fontsize=FONT.title, color=EDGE.blue)

    save(fig, "vae-reparameterization.png")


def main():
    figure_architecture()
    figure_reparameterization()


if __name__ == "__main__":
    main()
