import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colors import PALETTE

# Output directory (repo_root/public/images/notes), resolved relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(REPO_ROOT, "public", "images", "notes")

DPI = 200
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 13,
        "text.color": PALETTE["gray_dark"],
        "axes.edgecolor": PALETTE["gray_point"],
        "axes.labelcolor": PALETTE["gray_dark"],
        "xtick.color": PALETTE["gray_dark"],
        "ytick.color": PALETTE["gray_dark"],
        "figure.facecolor": PALETTE["background"],
        "axes.facecolor": PALETTE["background"],
        "savefig.facecolor": PALETTE["background"],
    }
)


def _clean_axes(ax):
    """Light, minimal axes that match the existing note figures."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["gray_point"])
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 1. Sigmoid / logistic curve
# ---------------------------------------------------------------------------
def figure_sigmoid():
    z = np.linspace(-8, 8, 400)
    g = 1.0 / (1.0 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _clean_axes(ax)

    # Asymptotes at 0 and 1
    for level in (0.0, 1.0):
        ax.axhline(
            level,
            color=PALETTE["gray_point"],
            linestyle=(0, (4, 3)),
            linewidth=1.1,
            zorder=1,
        )

    # Vertical reference at z = 0
    ax.axvline(
        0,
        color=PALETTE["gray_point"],
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        zorder=1,
    )

    # The sigmoid itself
    ax.plot(
        z,
        g,
        color=PALETTE["coral_dark"],
        linewidth=2.6,
        zorder=3,
        label=r"$g(z) = \dfrac{1}{1 + e^{-z}}$",
    )

    # Midpoint g(0) = 0.5
    ax.scatter(
        [0],
        [0.5],
        s=70,
        facecolor=PALETTE["blue_point"],
        edgecolor=PALETTE["blue_dark"],
        linewidth=1.4,
        zorder=4,
    )
    ax.annotate(
        r"$g(0) = \frac{1}{2}$",
        xy=(0, 0.5),
        xytext=(1.4, 0.36),
        fontsize=12,
        color=PALETTE["gray_dark"],
        arrowprops=dict(arrowstyle="-", color=PALETTE["gray_point"], linewidth=1.0),
    )

    ax.text(-7.7, 0.04, r"$0$", fontsize=12, color=PALETTE["gray_dark"], va="center")
    ax.text(-7.7, 0.96, r"$1$", fontsize=12, color=PALETTE["gray_dark"], va="center")

    ax.set_xlabel(r"$z = 0$")
    ax.set_ylabel(r"$h_\theta(x)$")
    ax.set_ylim(-0.08, 1.12)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.05),
        fontsize=13,
        handlelength=1.6,
        borderaxespad=0.6,
    )
    save(fig, "classification-sigmoid.png")


# ---------------------------------------------------------------------------
# 2. Binary decision boundary with shaded regions
# ---------------------------------------------------------------------------
def figure_decision_boundary():
    rng = np.random.default_rng(11)
    n = 22

    # Two roughly linearly separable clusters
    c0 = rng.normal(loc=[-1.4, -1.0], scale=[1.0, 1.0], size=(n, 2))
    c1 = rng.normal(loc=[1.6, 1.5], scale=[1.0, 1.0], size=(n, 2))

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    _clean_axes(ax)

    lim = 5.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # A fixed boundary theta^T x = 0 with theta = (theta0, theta1, theta2).
    # Boundary: t0 + t1*x1 + t2*x2 = 0.
    t0, t1, t2 = -0.3, 1.0, 1.1

    gx, gy = np.meshgrid(
        np.linspace(-lim, lim, 400),
        np.linspace(-lim, lim, 400),
    )
    score = t0 + t1 * gx + t2 * gy

    # Shade the two half-spaces (h > 0.5 vs h < 0.5)
    ax.contourf(
        gx,
        gy,
        score,
        levels=[0, score.max()],
        colors=[PALETTE["light_coral"]],
        alpha=0.55,
        zorder=0,
    )
    ax.contourf(
        gx,
        gy,
        score,
        levels=[score.min(), 0],
        colors=[PALETTE["light_blue"]],
        alpha=0.55,
        zorder=0,
    )

    # Decision boundary line
    xs = np.linspace(-lim, lim, 100)
    ys = -(t0 + t1 * xs) / t2
    ax.plot(
        xs,
        ys,
        color=PALETTE["gray_dark"],
        linewidth=2.0,
        zorder=2,
    )

    # Data points
    ax.scatter(
        c0[:, 0],
        c0[:, 1],
        s=68,
        facecolor=PALETTE["blue_point"],
        edgecolor=PALETTE["blue_dark"],
        linewidth=1.4,
        zorder=3,
        label=r"$y^{(i)} = 0$",
    )
    ax.scatter(
        c1[:, 0],
        c1[:, 1],
        s=68,
        facecolor=PALETTE["coral_point"],
        edgecolor=PALETTE["coral_dark"],
        linewidth=1.4,
        zorder=3,
        label=r"$y^{(i)} = 1$",
    )

    # Region annotations
    ax.text(
        2.7,
        3.9,
        r"$h_\theta(x) > \frac{1}{2}$",
        fontsize=12,
        color=PALETTE["coral_dark"],
        ha="center",
    )
    ax.text(
        -3.0,
        -3.9,
        r"$h_\theta(x) < \frac{1}{2}$",
        fontsize=12,
        color=PALETTE["blue_dark"],
        ha="center",
    )

    # Label the boundary, sitting just off the line and rotated to match it
    boundary_angle = np.degrees(np.arctan2(-(t1 / t2), 1.0))
    ax.text(
        xs[70] + 0.45,
        ys[70] + 0.55,
        r"$\theta^T x = 0$",
        fontsize=12,
        color=PALETTE["gray_dark"],
        ha="center",
        va="center",
        rotation=boundary_angle,
        rotation_mode="anchor",
        transform_rotates_text=True,
        zorder=4,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    legend = ax.legend(
        loc="upper left",
        fontsize=12,
        handlelength=1.0,
        borderaxespad=0.6,
        framealpha=1.0,
    )
    legend.get_frame().set_facecolor(PALETTE["background"])
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(5)
    save(fig, "classification-decision-boundary.png")


def main():
    figure_sigmoid()
    figure_decision_boundary()


if __name__ == "__main__":
    main()
