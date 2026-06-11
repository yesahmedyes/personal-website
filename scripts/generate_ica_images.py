import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from style import (
    setup, clean_axes, save,
    INK, AXIS, POINT, EDGE, TINT,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# 1. The cocktail-party / source-separation pipeline
# ---------------------------------------------------------------------------
def figure_cocktail_party():
    """Independent sources s are linearly mixed by A into the observed signals x;
    ICA learns the unmixing W = A^{-1} that recovers the sources."""
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    # Two clearly different source signals: a sine and a sawtooth.
    t = np.linspace(0, 1, 400)
    s1 = np.sin(2 * np.pi * 3 * t)
    saw_f = 2.0
    s2 = 2 * (saw_f * t - np.floor(saw_f * t + 0.5))  # sawtooth in [-1, 1]
    A = np.array([[0.75, 0.55], [0.45, 0.85]])
    x1 = A[0, 0] * s1 + A[0, 1] * s2
    x2 = A[1, 0] * s1 + A[1, 1] * s2

    CY_TOP, CY_BOT = 4.2, 1.8
    W_WAVE, H_WAVE = 1.5, 1.2

    def wave(cx, cy, y, color):
        tt = np.linspace(-W_WAVE / 2, W_WAVE / 2, len(y))
        yy = y / np.max(np.abs(y)) * (H_WAVE / 2) * 0.9
        ax.plot(cx + tt, cy + yy, color=color, linewidth=LW.line, zorder=3)

    def box(cx, cy, letter, face, edge):
        w, h = 1.0, 1.3
        ax.add_patch(
            Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=face,
                      edgecolor=edge, linewidth=LW.guide, zorder=2)
        )
        ax.text(cx, cy, letter, ha="center", va="center", fontsize=FONT.emphasis,
                color=INK, zorder=4)

    def arrow(p, q):
        ax.annotate("", xy=q, xytext=p, zorder=1, arrowprops=dict(
            arrowstyle="-|>", color=INK, linewidth=LW.guide,
            mutation_scale=15, shrinkA=3, shrinkB=3))

    CX_S, CX_A, CX_X, CX_W, CX_R = 1.25, 3.5, 5.9, 8.25, 10.6

    # Sources (blue = s1, coral = s2); mixtures in grey; recovered match sources.
    wave(CX_S, CY_TOP, s1, EDGE.blue)
    wave(CX_S, CY_BOT, s2, EDGE.coral)
    wave(CX_X, CY_TOP, x1, INK)
    wave(CX_X, CY_BOT, x2, INK)
    wave(CX_R, CY_TOP, s1, EDGE.blue)
    wave(CX_R, CY_BOT, s2, EDGE.coral)

    box(CX_A, 3.0, "$A$", TINT.coral, EDGE.coral)
    box(CX_W, 3.0, "$W$", TINT.blue, EDGE.blue)

    # Fan in to A, fan out to mixtures, fan in to W, fan out to recovered.
    arrow((CX_S + W_WAVE / 2, CY_TOP), (CX_A - 0.5, 3.2))
    arrow((CX_S + W_WAVE / 2, CY_BOT), (CX_A - 0.5, 2.8))
    arrow((CX_A + 0.5, 3.2), (CX_X - W_WAVE / 2, CY_TOP))
    arrow((CX_A + 0.5, 2.8), (CX_X - W_WAVE / 2, CY_BOT))
    arrow((CX_X + W_WAVE / 2, CY_TOP), (CX_W - 0.5, 3.2))
    arrow((CX_X + W_WAVE / 2, CY_BOT), (CX_W - 0.5, 2.8))
    arrow((CX_W + 0.5, 3.2), (CX_R - W_WAVE / 2, CY_TOP))
    arrow((CX_W + 0.5, 2.8), (CX_R - W_WAVE / 2, CY_BOT))

    # Per-wave labels above each waveform.
    lab = dict(ha="center", va="center", fontsize=FONT.title)
    ax.text(CX_S, CY_TOP + 0.92, "$s_1$", color=EDGE.blue, **lab)
    ax.text(CX_S, CY_BOT + 0.92, "$s_2$", color=EDGE.coral, **lab)
    ax.text(CX_X, CY_TOP + 0.92, "$x_1$", color=INK, **lab)
    ax.text(CX_X, CY_BOT + 0.92, "$x_2$", color=INK, **lab)
    ax.text(CX_R, CY_TOP + 0.92, r"$\hat{s}_1$", color=EDGE.blue, **lab)
    ax.text(CX_R, CY_BOT + 0.92, r"$\hat{s}_2$", color=EDGE.coral, **lab)

    # Column captions along the bottom.
    cap = dict(ha="center", va="center", fontsize=FONT.emphasis, color=INK)
    ax.text(CX_S, 0.5, "sources $s$", **cap)
    ax.text(CX_X, 0.5, "observed $x = As$", **cap)
    ax.text(CX_R, 0.5, r"recovered $\hat{s} = Wx$", **cap)

    save(fig, "ica-cocktail-party.png")


# ---------------------------------------------------------------------------
# 2. Mixing geometry: independent axes sheared into the columns of A
# ---------------------------------------------------------------------------
def figure_mixing_geometry():
    """Independent sources fill an axis-aligned square; the mixing matrix A maps
    that square to a parallelogram, sending the source axes to the columns of A."""
    rng = np.random.default_rng(3)
    S = rng.uniform(-1, 1, size=(300, 2))
    A = np.array([[1.0, 0.7], [0.35, 1.0]])
    X = S @ A.T

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))

    def panel(ax, P, basis, labels, title, lim):
        clean_axes(ax)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.scatter(P[:, 0], P[:, 1], s=MARK.small, facecolor=POINT.blue,
                   edgecolor=EDGE.blue, linewidth=LW.edge,
                   zorder=2)
        colors = [EDGE.coral, EDGE.green]
        for vec, lbl, c in zip(basis, labels, colors):
            ax.annotate("", xy=vec, xytext=(0, 0), zorder=4, arrowprops=dict(
                arrowstyle="-|>", color=c, linewidth=LW.arrow, mutation_scale=18,
                shrinkA=0, shrinkB=0))
            ax.text(vec[0] * 1.18, vec[1] * 1.18, lbl, color=c, fontsize=FONT.emphasis,
                    ha="center", va="center", zorder=5)
        ax.set_title(title, fontsize=FONT.title, color=INK)

    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    panel(axes[0], S, [e1, e2], [r"$e_1$", r"$e_2$"], "sources $s$", 1.5)
    panel(axes[1], X, [A @ e1, A @ e2], [r"$a_1$", r"$a_2$"],
          "observed $x = As$", 2.05)

    fig.subplots_adjust(wspace=0.1)
    save(fig, "ica-mixing-geometry.png")


# ---------------------------------------------------------------------------
# 3. Why Gaussian sources are unidentifiable
# ---------------------------------------------------------------------------
def figure_gaussian_ambiguity():
    """Non-Gaussian sources leave visible edges that fix the unmixing directions;
    an isotropic Gaussian cloud is rotationally symmetric, so the axes are not
    identifiable -- any rotation fits equally well."""
    rng = np.random.default_rng(7)
    N = 300
    A = np.array([[1.0, 0.7], [0.35, 1.0]])

    # Left: uniform (non-Gaussian) sources mixed -> parallelogram.
    S = rng.uniform(-1, 1, size=(N, 2))
    X_uni = S @ A.T

    # Right: isotropic Gaussian sources -> rotationally symmetric disc. Scaled to
    # roughly match the spatial extent of the left cloud so the panels balance.
    G = rng.normal(size=(N, 2)) * 0.55

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))

    def base(ax, P, lim):
        clean_axes(ax)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.scatter(P[:, 0], P[:, 1], s=MARK.small, facecolor=POINT.blue,
                   edgecolor=EDGE.blue, linewidth=LW.edge,
                   zorder=2)

    def axis_pair(ax, vecs, color, style, scale, labels=None):
        for k, v in enumerate(vecs):
            v = v / np.linalg.norm(v) * scale
            ax.annotate("", xy=v, xytext=(0, 0), zorder=4, arrowprops=dict(
                arrowstyle="-|>", color=color, linewidth=LW.arrow, mutation_scale=16,
                shrinkA=0, shrinkB=0, linestyle=style))
            if labels:
                ax.text(v[0] * 1.2, v[1] * 1.2, labels[k], color=color,
                        fontsize=FONT.emphasis, ha="center", va="center", zorder=5)

    # Left panel: recoverable axes aligned with the parallelogram edges.
    base(axes[0], X_uni, 2.0)
    axis_pair(axes[0], [A @ np.array([1.0, 0.0]), A @ np.array([0.0, 1.0])],
              EDGE.coral, "solid", 1.7, [r"$a_1$", r"$a_2$"])
    axes[0].set_title("Non-Gaussian: axes recoverable", fontsize=FONT.title,
                      color=INK)

    # Right panel: a round Gaussian disc with two different candidate frames,
    # both equally consistent with the rotational symmetry.
    base(axes[1], G, 2.0)
    axes[1].add_patch(Circle((0, 0), 1.7, facecolor="none",
                             edgecolor=AXIS,
                             linestyle=(0, (4, 3)), linewidth=LW.guide, zorder=1))
    th = np.radians(38)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    axis_pair(axes[1], [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
              EDGE.coral, "solid", 1.7)
    axis_pair(axes[1], [rot @ np.array([1.0, 0.0]), rot @ np.array([0.0, 1.0])],
              EDGE.green, (0, (3, 2)), 1.7)
    axes[1].set_title("Gaussian: axes ambiguous", fontsize=FONT.title,
                      color=INK)

    fig.subplots_adjust(wspace=0.1)
    save(fig, "ica-gaussian-ambiguity.png")


def main():
    figure_cocktail_party()
    figure_mixing_geometry()
    figure_gaussian_ambiguity()


if __name__ == "__main__":
    main()
