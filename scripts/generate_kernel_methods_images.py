import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (
    setup, clean_axes, save,
    INK, POINT, EDGE,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# 1. Feature map: non-separable in input space -> separable in feature space
# ---------------------------------------------------------------------------
def figure_feature_map():
    """Two concentric classes need a circular boundary in input space, but the
    quadratic feature map (x1, x2) -> (x1^2, x2^2) turns that boundary into a line."""
    rng = np.random.default_rng(1)
    n_in, n_out = 70, 130
    r_in = 1.05 * np.sqrt(rng.uniform(0, 1, n_in))
    th_in = rng.uniform(0, 2 * np.pi, n_in)
    Xin = np.column_stack([r_in * np.cos(th_in), r_in * np.sin(th_in)])
    r_out = rng.uniform(1.55, 2.35, n_out)
    th_out = rng.uniform(0, 2 * np.pi, n_out)
    Xout = np.column_stack([r_out * np.cos(th_out), r_out * np.sin(th_out)])
    R = 1.3  # boundary radius

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.0))

    def scatter(ax, P, fill, edge):
        ax.scatter(P[:, 0], P[:, 1], s=MARK.small, facecolor=fill, edgecolor=edge,
                   linewidth=LW.edge, zorder=3)

    # ----- input space -----
    ax = axes[0]
    clean_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.7, 2.7)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(th), R * np.sin(th), color=INK,
            linestyle=(0, (4, 3)), linewidth=LW.line, zorder=2)
    scatter(ax, Xout, POINT.blue, EDGE.blue)
    scatter(ax, Xin, POINT.coral, EDGE.coral)
    ax.set_title("input space", fontsize=FONT.emphasis, color=INK)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # ----- feature space -----
    ax = axes[1]
    clean_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 5.9)
    ax.set_ylim(-0.3, 5.9)
    # the same boundary x1^2 + x2^2 = R^2 is now the straight line phi1 + phi2 = R^2
    ax.plot([R ** 2 + 0.3, -0.3], [-0.3, R ** 2 + 0.3], color=INK,
            linestyle=(0, (4, 3)), linewidth=LW.line, zorder=2)
    scatter(ax, Xout ** 2, POINT.blue, EDGE.blue)
    scatter(ax, Xin ** 2, POINT.coral, EDGE.coral)
    ax.set_title("feature space", fontsize=FONT.emphasis, color=INK)
    ax.set_xlabel(r"$x_1^2$")
    ax.set_ylabel(r"$x_2^2$")

    fig.subplots_adjust(wspace=0.18)
    save(fig, "kernel-methods-feature-map.png")


def main():
    figure_feature_map()


if __name__ == "__main__":
    main()
