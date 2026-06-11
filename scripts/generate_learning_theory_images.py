import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from style import (
    setup, clean_axes, save,
    INK, AXIS, BACKGROUND, HUE, EDGE, POINT, TINT,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# 1. The bias-variance tradeoff curve
# ---------------------------------------------------------------------------
def figure_bias_variance():
    """Total error = bias^2 + variance: bias falls and variance rises with model
    complexity. The curves are mirror images, so they cross exactly at the optimum
    where the total error bottoms out."""
    c = 5.0  # optimum: bias^2 and variance cross here and the total is minimized
    x = np.linspace(0.6, 9.4, 400)
    bias2 = 0.4 + 0.45 * np.exp(-0.5 * (x - c))
    variance = 0.4 + 0.45 * np.exp(0.5 * (x - c))
    total = bias2 + variance

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    clean_axes(ax)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 5.6)

    ax.axvline(c, color=AXIS, linestyle=(0, (4, 3)),
               linewidth=LW.guide, zorder=1)

    ax.plot(x, total, color=INK, linewidth=LW.line, zorder=4)
    ax.plot(x, variance, color=EDGE.blue, linewidth=LW.line, zorder=3)
    ax.plot(x, bias2, color=EDGE.coral, linewidth=LW.line, zorder=3)

    ax.text(2.3, 4.15, "Total Error", color=INK, fontsize=FONT.emphasis, ha="center")
    ax.text(9.0, 2.0, "Variance", color=EDGE.blue, fontsize=FONT.emphasis, ha="center")
    ax.text(8.8, 0.75, "Bias$^2$", color=EDGE.coral, fontsize=FONT.emphasis, ha="center")
    ax.text(c + 0.20, 3.8, "optimum complexity", color=INK,
            fontsize=FONT.tick, rotation=90, ha="left", va="center")

    ax.set_xlabel("model complexity")
    ax.set_ylabel("error")
    save(fig, "learning-theory-bias-variance.png")


# ---------------------------------------------------------------------------
# 2. The bias-variance bullseye
# ---------------------------------------------------------------------------
def figure_bullseye():
    """The classic dartboard: bias is how far the cluster sits from the centre,
    variance is how spread out the shots are."""
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))

    rings = [
        (1.00, TINT.blue),
        (0.66, BACKGROUND),
        (0.40, HUE.blue),
        (0.14, EDGE.coral),
    ]
    offset = np.array([0.1, -0.5])           # systematic miss for the high-bias row
    specs = {
        (0, 0): (np.array([0.0, 0.0]), 0.075, 1),
        (0, 1): (np.array([0.0, 0.0]), 0.27, 2),
        (1, 0): (offset, 0.075, 3),
        (1, 1): (offset, 0.27, 4),
    }

    for (r, c), (mean, sd, seed) in specs.items():
        ax = axes[r, c]
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)

        for rad, col in rings:
            ax.add_patch(Circle((0, 0), rad, facecolor=col, edgecolor="none", zorder=1))
        ax.add_patch(Circle((0, 0), 1.0, facecolor="none",
                            edgecolor=EDGE.blue, linewidth=LW.guide, zorder=2))

        rng = np.random.default_rng(seed)
        pts = rng.normal(mean, sd, size=(11, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=MARK.small, facecolor=INK,
                   edgecolor=BACKGROUND, linewidth=LW.edge, zorder=4)

    axes[0, 0].set_title("Low Variance", fontsize=FONT.emphasis, color=INK)
    axes[0, 1].set_title("High Variance", fontsize=FONT.emphasis, color=INK)
    axes[0, 0].set_ylabel("Low Bias", fontsize=FONT.emphasis, color=INK)
    axes[1, 0].set_ylabel("High Bias", fontsize=FONT.emphasis, color=INK)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.04,
                        wspace=0.05, hspace=0.05)
    save(fig, "learning-theory-bullseye.png")


# ---------------------------------------------------------------------------
# 3. The double-descent curve
# ---------------------------------------------------------------------------
def figure_double_descent():
    """Classical U-shaped test error, a spike at the interpolation threshold where
    the parameter count matches the sample size, then a second descent."""
    x = np.linspace(0.3, 10, 600)
    t = 5.0
    base = 1.2 + 3.5 * np.exp(-0.7 * x)
    bump = 2.6 / (1 + ((x - t) / 0.7) ** 2)
    test = base + bump

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    clean_axes(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)

    ax.plot([t, t], [0, 4.15], color=AXIS, linestyle=(0, (4, 3)),
            linewidth=LW.guide, zorder=1)
    ax.plot(x, test, color=INK, linewidth=LW.line, zorder=3)

    ax.text(t, 4.42, r"interpolation threshold $d \approx n$",
            color=INK, fontsize=FONT.annotation, ha="center")
    ax.text(2.5, 0.5, "classical\nregime", color=INK, fontsize=FONT.annotation,
            ha="center", va="center", linespacing=1.4)
    ax.text(7.7, 0.5, "over-parameterized\nregime", color=INK,
            fontsize=FONT.annotation, ha="center", va="center", linespacing=1.4)
    ax.text(1.6, 2.9, "test error", color=INK, fontsize=FONT.emphasis)

    ax.set_xlabel(r"number of parameters $d$")
    ax.set_ylabel("test error")
    save(fig, "learning-theory-double-descent.png")


def main():
    figure_bias_variance()
    figure_bullseye()
    figure_double_descent()


if __name__ == "__main__":
    main()
