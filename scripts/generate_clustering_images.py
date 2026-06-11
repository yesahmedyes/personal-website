import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from style import (
    setup, clean_axes, save,
    INK, AXIS, BACKGROUND,
    POINT, POINTS, EDGES, REGIONS,
    FONT, LW, MARK,
)

setup()


def arrow(ax, p, q, color=INK, lw=LW.line):
    ax.annotate("", xy=q, xytext=p, zorder=7, arrowprops=dict(
        arrowstyle="->", color=color, linewidth=lw, mutation_scale=12,
        shrinkA=6, shrinkB=6))


def fig_arrow(fig, p, q, rad=0.0, lw=LW.line, mscale=16, color=INK):
    """An arrow drawn in figure coordinates, used to connect the panels."""
    ar = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=mscale, linewidth=lw,
                         color=color, shrinkA=0, shrinkB=0, zorder=10,
                         connectionstyle=f"arc3,rad={rad}")
    ar.set_transform(fig.transFigure)
    fig.add_artist(ar)
    return ar


# ---------------------------------------------------------------------------
# Minimal k-means, so every panel shows an honest state of the algorithm.
# ---------------------------------------------------------------------------
def _assign(X, C):
    d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def _update(X, labels, C):
    return np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                     for j in range(len(C))])


def _distortion(X, C, labels):
    return float(((X - C[labels]) ** 2).sum())


def _run(X, C, n_iter=100):
    C = C.copy()
    for _ in range(n_iter):
        labels = _assign(X, C)
        new_C = _update(X, labels, C)
        if np.allclose(new_C, C):
            C = new_C
            break
        C = new_C
    labels = _assign(X, C)
    return C, labels, _distortion(X, C, labels)


def _centroids(ax, C, hollow=False, s=MARK.star):
    """Draw cluster centroids as starred markers in each cluster's dark colour."""
    for j, c in enumerate(C):
        ax.scatter(c[0], c[1], marker="*", s=s,
                   facecolor=(BACKGROUND if hollow else EDGES[j]),
                   edgecolor=EDGES[j] if hollow else INK,
                   linewidth=LW.guide, zorder=6)


def _voronoi(ax, C, xlim, ylim):
    gx, gy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    cls = _assign(grid, C).reshape(gx.shape).astype(float)
    ax.contourf(gx, gy, cls, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=REGIONS, zorder=0)


def _scatter(ax, X, labels=None, s=MARK.point):
    if labels is None:                       # unassigned: neutral grey
        ax.scatter(X[:, 0], X[:, 1], s=s, facecolor=POINT.muted,
                   edgecolor=INK, linewidth=LW.edge, zorder=3)
        return
    for j in range(int(labels.max()) + 1):
        m = labels == j
        ax.scatter(X[m, 0], X[m, 1], s=s, facecolor=POINTS[j], edgecolor=EDGES[j],
                   linewidth=LW.edge, zorder=3)


# ---------------------------------------------------------------------------
# 1. The k-means loop: initialise -> assign -> update -> converged.
# ---------------------------------------------------------------------------
def figure_iterations():
    """The k-means loop. Panel 1 drops in random centroids; panel 2 is the
    assignment step (each point to its nearest centroid); panel 3 is the update
    step (each centroid to the mean of its points). The two steps alternate until
    the assignment stops changing."""
    rng = np.random.default_rng(3)
    X = np.vstack([
        rng.normal([-2.6, 2.3], 0.95, size=(40, 2)),
        rng.normal([2.7, 1.9], 0.95, size=(40, 2)),
        rng.normal([0.1, -2.5], 0.95, size=(40, 2)),
    ])

    C0 = np.array([[-1.6, 0.4], [1.4, 1.0], [-0.2, -1.1]])
    labels0 = _assign(X, C0)
    C1 = _update(X, labels0, C0)

    pad = 1.0
    cx = 0.5 * (X[:, 0].min() + X[:, 0].max())
    cy = 0.5 * (X[:, 1].min() + X[:, 1].max())
    half = 0.5 * max(np.ptp(X[:, 0]), np.ptp(X[:, 1])) + pad
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    # Explicit square panels, so figure-coordinate arrows land where intended.
    Wfig, Hfig = 10.6, 4.9
    fig = plt.figure(figsize=(Wfig, Hfig))
    y0, h = 0.33, 0.55
    pw = h * Hfig / Wfig                 # panel width that renders square
    g = 0.082                            # gap between panels
    x_start = (1.0 - (3 * pw + 2 * g)) / 2.0
    x0s = [x_start + i * (pw + g) for i in range(3)]

    titles = ["initialize centroids", "assign points", "update centroids"]
    axes = []
    for i in range(3):
        ax = fig.add_axes([x0s[i], y0, pw, h])
        clean_axes(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(titles[i], fontsize=FONT.title, color=INK, pad=8)
        axes.append(ax)

    # Panel 1: grey points, centroids dropped in at random.
    _scatter(axes[0], X, None)
    _centroids(axes[0], C0)

    # Panel 2: colour every point by its nearest centroid (the first inner loop).
    _voronoi(axes[1], C0, xlim, ylim)
    _scatter(axes[1], X, labels0)
    _centroids(axes[1], C0)

    # Panel 3: each centroid slides to the mean of its points (the second loop).
    _voronoi(axes[2], C0, xlim, ylim)
    _scatter(axes[2], X, labels0)
    _centroids(axes[2], C0, hollow=True)
    for old, new in zip(C0, C1):
        arrow(axes[2], old, new)
    _centroids(axes[2], C1)

    # Forward flow: initialise -> assign -> update, in the gaps at mid-height,
    # kept clear of the panel edges.
    yc = y0 + h / 2
    inset = 0.017
    for i in range(2):
        fig_arrow(fig, (x0s[i] + pw + inset, yc), (x0s[i + 1] - inset, yc))

    # Loop-back: update -> assign, a shallow bow just below the panels (rad < 0).
    cxr, cxl = x0s[2] + pw / 2, x0s[1] + pw / 2
    yb = y0 - 0.003
    fig_arrow(fig, (cxr, yb), (cxl, yb), rad=-0.20)
    fig.text((cxl + cxr) / 2, y0 - 0.1, "repeat → converged", ha="center",
             va="top", fontsize=FONT.annotation, color=INK)

    save(fig, "clustering-kmeans-iterations.png")


# ---------------------------------------------------------------------------
# 2. Initialisation matters: the same data can land at very different optima.
# ---------------------------------------------------------------------------
def figure_local_minima():
    """k-means minimises a non-convex distortion J, so where it ends up depends on
    where it starts. A lucky start recovers the three clusters (low J); an unlucky
    one splits one blob and merges two others into a stable but worse optimum."""
    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal([-3.3, 1.9], 0.72, size=(35, 2)),
        rng.normal([3.3, 1.9], 0.72, size=(35, 2)),
        rng.normal([0.0, -2.9], 0.72, size=(35, 2)),
    ])

    good_init = np.array([[-3.0, 1.9], [3.0, 1.9], [0.0, -2.9]])
    bad_init = np.array([[-3.5, 2.4], [-2.9, 1.3], [1.7, -0.5]])

    Cg, lg, Jg = _run(X, good_init)
    Cb, lb, Jb = _run(X, bad_init)
    print(f"  good J = {Jg:.1f}   bad J = {Jb:.1f}")

    pad = 1.1
    xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
    ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9))
    panels = [
        (axes[0], Cg, lg, Jg, "lucky start"),
        (axes[1], Cb, lb, Jb, "unlucky start"),
    ]
    for ax, C, labels, J, title in panels:
        clean_axes(ax)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=FONT.title, color=INK, pad=8)
        _voronoi(ax, C, xlim, ylim)
        _scatter(ax, X, labels)
        _centroids(ax, C)
        ax.text(0.5, 0.05, fr"$J \approx {J:.0f}$", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=FONT.annotation, color=INK,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=BACKGROUND,
                          edgecolor=AXIS, linewidth=LW.edge))

    fig.subplots_adjust(wspace=0.16)
    save(fig, "clustering-local-minima.png")


def main():
    figure_iterations()
    figure_local_minima()


if __name__ == "__main__":
    main()
