import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import (
    setup, clean_axes_off, save,
    INK, AXIS, BACKGROUND,
    HUE, EDGE, TINT,
    FONT, LW, MARK,
)

setup()


def arrow(ax, p, q, color=INK, lw=LW.line, rad=0.0, ls="solid"):
    ax.annotate("", xy=q, xytext=p, zorder=2, arrowprops=dict(
        arrowstyle="-|>", color=color, linewidth=lw, mutation_scale=15,
        shrinkA=2, shrinkB=2, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------------------
# The EM lower-bound picture: ELBO touches l(theta) at theta^(t), and
# maximizing it lands at theta^(t+1), where l is higher still.
# ---------------------------------------------------------------------------
def figure_lower_bound():
    """One EM iteration. The E step makes ELBO(x; Q^t, .) tangent to the
    log-likelihood l at theta^t (A); the M step maximizes that bound to reach
    theta^{t+1} (B); and since l >= ELBO always, l(theta^{t+1}) (C) is higher
    still -- so the iteration cannot decrease the likelihood."""

    def ell_fn(t):
        return (0.9 / (1 + np.exp(-3.0 * (t - 2.0)))
                + 4.6 / (1 + np.exp(-1.15 * (t - 5.0)))
                - 0.035 * (t - 1.0) ** 2 + 0.25)

    th_t, th_next = 4.4, 6.0
    ell_t = ell_fn(th_t)
    slope = (ell_fn(th_t + 1e-3) - ell_fn(th_t - 1e-3)) / 2e-3  # l'(theta^t)
    a = slope / (2 * (th_next - th_t))                          # peak at th_next

    theta = np.linspace(0.9, 9.7, 700)
    ell = ell_fn(theta)
    elbo = ell_t + slope * (theta - th_t) - a * (theta - th_t) ** 2

    A = (th_t, ell_t)
    B = (th_next, ell_t + slope * (th_next - th_t) - a * (th_next - th_t) ** 2)
    C = (th_next, ell_fn(th_next))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    clean_axes_off(ax)
    ax.set_xlim(-0.4, 10.8)
    ax.set_ylim(-0.85, 4.8)

    # Coordinate axes as arrows meeting at the origin.
    axis_arrow = dict(arrowstyle="-|>", color=INK, linewidth=LW.guide,
                      mutation_scale=18, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(10.5, 0), xytext=(0, 0), arrowprops=axis_arrow, zorder=2)
    ax.annotate("", xy=(0, 4.6), xytext=(0, 0), arrowprops=axis_arrow, zorder=2)

    # The two curves; the ELBO is shown only where it stays above the axis.
    ax.plot(theta, ell, color=INK, linewidth=LW.line, zorder=3)
    ax.plot(theta, np.where(elbo >= 0.5, elbo, np.nan), color=EDGE.blue,
            linewidth=LW.line, zorder=3)

    # Curve labels.
    ax.text(9.0, ell_fn(9.0) + 0.22, r"$\ell(\theta)$", fontsize=FONT.title,
            color=INK)
    ax.text(8.0, 2.0, r"$\mathrm{ELBO}(x;\, Q^{(t)},\, \theta)$", fontsize=FONT.title,
            color=EDGE.blue)

    # Dashed drop-lines at the two iterates.
    for x, top in ((th_t, A[1]), (th_next, C[1])):
        ax.plot([x, x], [0, top], color=AXIS,
                linestyle=(0, (4, 3)), linewidth=LW.guide, zorder=1)

    # Points A, B, C.
    for (px, py), label, (lx, ly) in (
        (A, "A", (-0.23, 0.22)),
        (B, "B", (0.25, 0.21)),
        (C, "C", (-0.20, 0.20)),
    ):
        ax.scatter([px], [py], s=MARK.small, color=INK, zorder=5)
        ax.text(px + lx, py + ly, label, fontsize=FONT.title, color=INK,
                ha="center", va="center")

    # Axis ticks for the two iterates.
    ax.text(th_t, -0.5, r"$\theta^{(t)}$", ha="center", va="center", fontsize=FONT.title)
    ax.text(th_next, -0.5, r"$\theta^{(t+1)}$", ha="center", va="center",
            fontsize=FONT.title)

    save(fig, "em-lower-bound.png")


# ---------------------------------------------------------------------------
# The E-step view: at fixed theta, choosing Q = p(z | x) closes the KL gap.
# ---------------------------------------------------------------------------
def figure_elbo_gap():
    """For a fixed theta, log p(x; theta) splits into the ELBO plus the KL gap
    between Q and the true posterior. The E step sets Q = p(z | x), making the
    gap exactly zero, so the bound is tight: ELBO = log p(x; theta)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.45))
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.7, 5.9)

    T = 5.0          # log p(x; theta)
    W = 1.0
    bars = {1.4: 3.0, 3.6: 5.0}   # x-center -> ELBO height (KL = T - ELBO)

    for cx, elbo in bars.items():
        ax.add_patch(Rectangle((cx - W / 2, 0), W, elbo,
                              facecolor=HUE.blue,
                              edgecolor=EDGE.blue, linewidth=LW.line,
                              zorder=3))
        if T - elbo > 1e-6:   # no KL box once the gap is exactly closed
            ax.add_patch(Rectangle((cx - W / 2, elbo), W, T - elbo,
                                  facecolor=TINT.coral,
                                  edgecolor=EDGE.coral, linewidth=LW.line,
                                  zorder=3))
        ax.text(cx, elbo / 2, "ELBO", ha="center", va="center", fontsize=FONT.annotation,
                color=INK, zorder=4)

    # Fixed ceiling at log p(x).
    ax.plot([0.45, 4.65], [T, T], color=AXIS,
            linestyle=(0, (5, 3)), linewidth=LW.guide, zorder=2)
    ax.text(2.5, T + 0.5, r"$\log p(x; \theta)$", ha="center",
            va="bottom", fontsize=FONT.annotation, color=INK)

    # Label the KL gap directly to the left of the red box.
    ax.text(0.78, 4.0, r"$D_{KL}(Q \,\|\, p_{z|x})$", ha="right", va="center",
            fontsize=FONT.annotation, color=EDGE.coral)
    ax.text(3.6, T + 0.12, r"$D_{KL} = 0$", ha="center", va="bottom",
            fontsize=FONT.tick, color=EDGE.coral)

    # The E-step jump between the two bars.
    arrow(ax, (1.95, 3.4), (3.05, 3.4), rad=-0.3, lw=LW.line)
    ax.text(2.5, 4.2, "maximize\nELBO", ha="center", va="center",
            fontsize=FONT.annotation, color=INK, linespacing=1.3)

    # Bar captions.
    ax.text(1.4, -0.45, "initial $Q$", ha="center", va="center", fontsize=FONT.annotation,
            color=INK)
    ax.text(3.6, -0.45, r"$Q = p(z \mid x)$", ha="center", va="center",
            fontsize=FONT.annotation, color=INK)

    save(fig, "em-elbo-gap.png")


# ---------------------------------------------------------------------------
# The EM staircase: successive tangent bounds climb l(theta) to a local max.
# ---------------------------------------------------------------------------
def figure_staircase():
    """Each iteration builds a lower bound tangent to l at theta^(k) (E step) and
    maximizes it (M step), so theta^(k) steps up the likelihood. The steps shrink
    as l flattens, converging to a local maximum."""
    def ell_fn(t):
        return 5.2 / (1 + np.exp(-0.9 * (t - 3.5))) - 0.03 * (t - 1.0) ** 2 + 0.3

    def ellp(t):
        return (ell_fn(t + 1e-3) - ell_fn(t - 1e-3)) / 2e-3

    a = 0.28
    thetas = [2.0]
    for _ in range(3):
        tk = thetas[-1]
        thetas.append(tk + ellp(tk) / (2 * a))   # M-step peak of the bound

    theta = np.linspace(0.8, 10.5, 700)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    clean_axes_off(ax)
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylim(-0.75, 5.05)

    axis_arrow = dict(arrowstyle="-|>", color=INK, linewidth=LW.guide,
                      mutation_scale=18, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(11.0, 0), xytext=(0, 0), arrowprops=axis_arrow, zorder=2)
    ax.annotate("", xy=(0, 4.85), xytext=(0, 0), arrowprops=axis_arrow, zorder=2)
    ax.text(11.15, 0, r"$\theta$", ha="left", va="center", fontsize=FONT.emphasis)

    ax.plot(theta, ell_fn(theta), color=INK, linewidth=LW.line,
            zorder=4)
    ax.text(9.0, ell_fn(9.2) + 0.30, r"$\ell(\theta)$", fontsize=FONT.emphasis,
            color=INK)

    # Successive tangent lower bounds, drawn as full concave parabolas that
    # touch l at theta^(k) and peak at the next iterate.
    for k in range(len(thetas) - 1):
        tk, tk1 = thetas[k], thetas[k + 1]
        m = ellp(tk)
        half = (tk1 - tk) + 0.6
        seg = np.linspace(tk1 - half, tk1 + half, 240)
        g = ell_fn(tk) + m * (seg - tk) - a * (seg - tk) ** 2
        ax.plot(seg, g, color=EDGE.blue, linewidth=LW.line, zorder=3)

    # Iterate dots climbing l(theta), with dashed read-offs to the axis.
    for k, tk in enumerate(thetas):
        yk = ell_fn(tk)
        ax.plot([tk, tk], [0, yk], color=AXIS,
                linestyle=(0, (4, 3)), linewidth=LW.guide, zorder=1)
        ax.scatter([tk], [yk], s=MARK.small, color=INK, zorder=6)
        ax.text(tk, -0.52, fr"$\theta^{{({k})}}$", ha="center", va="center",
                fontsize=FONT.title)

    save(fig, "em-staircase.png")


def main():
    figure_lower_bound()
    figure_elbo_gap()
    figure_staircase()


if __name__ == "__main__":
    main()
