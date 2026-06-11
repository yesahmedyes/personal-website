import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from style import (
    setup, clean_axes, save,
    INK, AXIS, BACKGROUND,
    POINT, EDGE, REGION,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# 1. Decision tree <-> axis-aligned partition of the feature space
# ---------------------------------------------------------------------------
def figure_partition():
    """A depth-2 tree and the rectangular regions its splits carve out; each leaf
    on the right corresponds to one shaded region on the left."""
    t1, t2, t3 = 0.5, 0.55, 0.35
    blue, coral = POINT.blue, POINT.coral
    bdk, cdk = EDGE.blue, EDGE.coral
    lblue, lcoral = REGION.blue, REGION.coral

    rng = np.random.default_rng(0)

    def pts(x0, x1, y0, y1, n):
        return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])

    # region -> (majority points, minority points)
    R1 = (pts(0.02, t1, 0.02, t2, 15), pts(0.02, t1, 0.02, t2, 2))            # blue
    R2 = (pts(0.02, t1, t2, 0.98, 13), pts(0.02, t1, t2, 0.98, 2))            # coral
    R3 = (pts(t1, 0.98, 0.02, t3, 9), pts(t1, 0.98, 0.02, t3, 1))             # coral
    R4 = (pts(t1, 0.98, t3, 0.98, 16), pts(t1, 0.98, t3, 0.98, 2))            # blue

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.9))

    # ----- left: feature space partition -----
    axL.set_aspect("equal")
    clean_axes(axL)
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    regions = [
        (0, 0, t1, t2, lblue),
        (0, t2, t1, 1, lcoral),
        (t1, 0, 1, t3, lcoral),
        (t1, t3, 1, 1, lblue),
    ]
    for x0, y0, x1, y1, col in regions:
        axL.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=col,
                                edgecolor="none", zorder=1))
    # split lines (only across the subregion they apply to)
    line = dict(color=INK, linewidth=LW.line, zorder=2)
    axL.plot([t1, t1], [0, 1], **line)
    axL.plot([0, t1], [t2, t2], **line)
    axL.plot([t1, 1], [t3, t3], **line)

    def scatter(P, fill, edge):
        axL.scatter(P[:, 0], P[:, 1], s=MARK.small, facecolor=fill, edgecolor=edge,
                    linewidth=LW.edge, zorder=3)

    for maj, fill, edge in [(R1, blue, bdk), (R4, blue, bdk)]:
        scatter(maj[0], blue, bdk)
        scatter(maj[1], coral, cdk)
    for maj in [R2, R3]:
        scatter(maj[0], coral, cdk)
        scatter(maj[1], blue, bdk)

    labpos = [(0.045, 0.49, "1"), (0.045, 0.93, "2"), (0.545, 0.29, "3"),
              (0.545, 0.93, "4")]
    for lx, ly, t in labpos:
        axL.text(lx, ly, t, fontsize=FONT.annotation, color=INK, ha="center",
                 va="center", zorder=4,
                 bbox=dict(boxstyle="circle,pad=0.18", facecolor=BACKGROUND,
                           edgecolor=AXIS, linewidth=LW.edge))
    axL.set_xlabel(r"$x_1$")
    axL.set_ylabel(r"$x_2$")
    axL.set_title("feature space", fontsize=FONT.title, color=INK)

    # ----- right: the tree -----
    axR.set_axis_off()
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 1)

    def node(cx, cy, text, w=0.26, h=0.14, fc=BACKGROUND,
             ec=INK, fs=FONT.annotation):
        axR.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                      boxstyle="round,pad=0.012,rounding_size=0.03", facecolor=fc,
                      edgecolor=ec, linewidth=LW.line, zorder=3))
        axR.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)

    def leaf(cx, cy, num, fc, ec):
        w = h = 0.13
        axR.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                      boxstyle="round,pad=0.012,rounding_size=0.03", facecolor=fc,
                      edgecolor=ec, linewidth=LW.line, zorder=3))
        axR.text(cx, cy, num, ha="center", va="center", fontsize=FONT.annotation,
                 color=INK, zorder=4)

    def edge(p, q, label):
        axR.plot([p[0], q[0]], [p[1], q[1]], color=AXIS,
                 linewidth=LW.line, zorder=1)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        axR.text(mx, my, label, fontsize=FONT.tick, color=INK,
                 ha="center", va="center", zorder=2,
                 bbox=dict(facecolor=BACKGROUND, edgecolor="none", pad=0.6))

    root, Lin, Rin = (0.5, 0.86), (0.28, 0.52), (0.72, 0.52)
    l1, l2, l3, l4 = (0.13, 0.16), (0.40, 0.16), (0.60, 0.16), (0.87, 0.16)
    edge(root, Lin, "yes")
    edge(root, Rin, "no")
    edge(Lin, l1, "yes")
    edge(Lin, l2, "no")
    edge(Rin, l3, "yes")
    edge(Rin, l4, "no")
    node(*root, r"$x_1 < 0.5$")
    node(*Lin, r"$x_2 < 0.55$")
    node(*Rin, r"$x_2 < 0.35$")
    leaf(*l1, "1", lblue, bdk)
    leaf(*l2, "2", lcoral, cdk)
    leaf(*l3, "3", lcoral, cdk)
    leaf(*l4, "4", lblue, bdk)

    fig.subplots_adjust(wspace=0.12)
    save(fig, "decision-trees-partition.png")


# ---------------------------------------------------------------------------
# 2. Gini vs entropy impurity curves
# ---------------------------------------------------------------------------
def figure_impurity():
    """Both impurities vanish for a pure node and peak at p = 1/2."""
    p = np.linspace(1e-6, 1 - 1e-6, 500)
    gini = 2 * p * (1 - p)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    clean_axes(ax)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(length=3, labelsize=FONT.tick)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)

    ax.axvline(0.5, color=AXIS, linestyle=(0, (4, 3)),
               linewidth=LW.guide, zorder=1)
    ax.plot(p, entropy, color=EDGE.blue, linewidth=LW.arrow, zorder=3)
    ax.plot(p, gini, color=EDGE.coral, linewidth=LW.arrow, zorder=3)

    ax.text(0.20, 0.90, "entropy", color=EDGE.blue, fontsize=FONT.emphasis, ha="center")
    ax.text(0.75, 0.45, "Gini", color=EDGE.coral, fontsize=FONT.emphasis, ha="center")

    ax.set_xlabel(r"class proportion $p$")
    ax.set_ylabel("impurity")
    save(fig, "decision-trees-impurity.png")


# ---------------------------------------------------------------------------
# Axis-aligned decision stump (for the AdaBoost figure)
# ---------------------------------------------------------------------------
def _stump_predict(X, feat, thr, pol):
    return np.where(X[:, feat] < thr, pol, -pol)


# ---------------------------------------------------------------------------
# 4. AdaBoost reweights the hard examples each round
# ---------------------------------------------------------------------------
def figure_adaboost():
    """The AdaBoost pipeline: a weak learner is fit, the points it misclassifies are
    upweighted, the next weak learner focuses on them, and the weighted combination
    of the weak learners forms the strong learner."""
    blue, coral = POINT.blue, POINT.coral
    bdk, cdk = EDGE.blue, EDGE.coral
    lblue, lcoral = REGION.blue, REGION.coral
    gdk, gpt = INK, AXIS

    # "corner" data: blue sits in the upper-right; coral fills the L-shaped rest. No
    # single axis-aligned stump separates it, but a vertical and a horizontal stump
    # together do -- so the two weak learners look clearly different.
    rng = np.random.default_rng(0)
    blue_pts = np.column_stack([rng.uniform(0.57, 0.90, 5), rng.uniform(0.57, 0.90, 5)])
    coral_l = np.column_stack([rng.uniform(0.10, 0.42, 4), rng.uniform(0.10, 0.90, 4)])
    coral_br = np.column_stack([rng.uniform(0.57, 0.90, 3), rng.uniform(0.10, 0.42, 3)])
    X = np.vstack([blue_pts, coral_l, coral_br])
    y = np.array([1] * 5 + [-1] * 7)
    n = len(y)

    stump1 = dict(feat=0, thr=0.5, pol=-1)          # vertical cut: x1 > 0.5 -> blue
    stump2 = dict(feat=1, thr=0.5, pol=-1)          # horizontal cut: x2 > 0.5 -> blue
    mis1 = np.where(_stump_predict(X, **stump1) != y)[0]   # bottom-right coral points
    base = np.full(n, 80.0)
    heavy = base.copy()
    heavy[mis1] = 340.0                              # upweight what weak learner #1 got wrong

    fig = plt.figure(figsize=(11.6, 6.0))

    pos = {
        "data":   [0.03, 0.58, 0.21, 0.31],
        "wdata":  [0.35, 0.58, 0.21, 0.31],
        "wl1":    [0.03, 0.12, 0.21, 0.31],
        "wl2":    [0.35, 0.12, 0.21, 0.31],
        "strong": [0.71, 0.34, 0.26, 0.33],
    }

    def panel(key):
        ax = fig.add_axes(pos[key])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(gpt)
            s.set_linewidth(LW.guide)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def pts(ax, sizes, ring=None):
        if ring is not None:
            ax.scatter(X[ring, 0], X[ring, 1], s=sizes[ring] + 150, facecolor="none",
                       edgecolor=gdk, linewidth=LW.line, zorder=6)
        for m, fill, edge in [(y == 1, blue, bdk), (y == -1, coral, cdk)]:
            ax.scatter(X[m, 0], X[m, 1], s=sizes[m], facecolor=fill, edgecolor=edge,
                       linewidth=LW.edge, zorder=5)

    def stump(ax, rd):
        feat, thr, pol = rd["feat"], rd["thr"], rd["pol"]
        c_lo = lblue if pol == 1 else lcoral
        c_hi = lcoral if pol == 1 else lblue
        if feat == 0:
            ax.axvspan(0, thr, color=c_lo, zorder=0)
            ax.axvspan(thr, 1, color=c_hi, zorder=0)
            ax.plot([thr, thr], [0, 1], color=gdk, linestyle="-.", linewidth=LW.line, zorder=2)
        else:
            ax.axhspan(0, thr, color=c_lo, zorder=0)
            ax.axhspan(thr, 1, color=c_hi, zorder=0)
            ax.plot([0, 1], [thr, thr], color=gdk, linestyle="-.", linewidth=LW.line, zorder=2)

    pts(panel("data"), base)
    a = panel("wl1"); stump(a, stump1); pts(a, base, ring=mis1)
    pts(panel("wdata"), heavy, ring=mis1)
    a = panel("wl2"); stump(a, stump2); pts(a, heavy)

    # strong learner: the corner formed by combining the vertical and horizontal cuts
    a = panel("strong")
    a.add_patch(Rectangle((0, 0), 0.5, 1.0, facecolor=lcoral, edgecolor="none", zorder=0))
    a.add_patch(Rectangle((0.5, 0), 0.5, 0.5, facecolor=lcoral, edgecolor="none", zorder=0))
    a.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, facecolor=lblue, edgecolor="none", zorder=0))
    a.plot([0.5, 0.5], [0.5, 1.0], color=gdk, linestyle="-.", linewidth=LW.line, zorder=2)
    a.plot([0.5, 1.0], [0.5, 0.5], color=gdk, linestyle="-.", linewidth=LW.line, zorder=2)
    pts(a, base)

    # labels under each panel (drawn on the figure so the tight bbox stays snug)
    lab = dict(va="top", fontsize=FONT.title, color=gdk)
    fig.text(0.06, 0.52, "Dataset", ha="left", **lab)
    fig.text(0.405, 0.535, "Weighted\ndataset", ha="center", linespacing=1.3, **lab)
    fig.text(0.135, 0.09, "Weak learner #1", ha="center", **lab)
    fig.text(0.455, 0.09, "Weak learner #2", ha="center", **lab)
    fig.text(0.84, 0.31, "Strong learner", ha="center", **lab)

    def arrow(p, q, lw=LW.line, mscale=16):
        ar = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=mscale, linewidth=lw,
                             color=gdk)
        ar.set_transform(fig.transFigure)
        fig.add_artist(ar)

    arrow((0.135, 0.58), (0.135, 0.43))     # dataset -> weak learner #1
    arrow((0.455, 0.58), (0.455, 0.43))     # weighted dataset -> weak learner #2

    # weak learner #1 -> weighted dataset: one stepped arrow routed up the gap
    step = Line2D([0.24, 0.30, 0.30], [0.27, 0.27, 0.73], color=gdk, linewidth=LW.line,
                  solid_joinstyle="round", solid_capstyle="round")
    step.set_transform(fig.transFigure)
    fig.add_artist(step)
    arrow((0.298, 0.73), (0.353, 0.73))

    # one bold arrow carries the boosted result through to the strong learner
    arrow((0.605, 0.505), (0.675, 0.505), lw=LW.arrow, mscale=28)

    save(fig, "decision-trees-adaboost.png")


def main():
    figure_partition()
    figure_impurity()
    figure_adaboost()


if __name__ == "__main__":
    main()
