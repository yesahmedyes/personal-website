import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from style import (
    setup, clean_axes, save,
    INK, AXIS, POINT, EDGE,
    FONT, LW, MARK,
)

setup()


def _principal_axes(X):
    """Return the mean, eigenvalues and eigenvectors (descending) of cov(X)."""
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Sigma)  # ascending
    order = np.argsort(eigvals)[::-1]
    return mu, eigvals[order], eigvecs[:, order]


# ---------------------------------------------------------------------------
# 1. Principal components drawn on the data cloud
# ---------------------------------------------------------------------------
def figure_principal_components():
    """The canonical PCA picture: eigenvectors of the covariance, scaled by the
    standard deviation along each axis, drawn over a correlated data cloud."""
    rng = np.random.default_rng(4)
    cov = np.array([[1.4, 0.9], [0.9, 1.0]])
    X = rng.multivariate_normal([0.0, 0.0], cov, size=170)

    scale = 2.2  # arrow length = scale * sqrt(eigenvalue); tips land on the ellipse
    label_offsets = [1.105, 1.38]  # multiples of the arrow length for the u labels

    # Drop the single point that would sit under the u2 label, then recompute the
    # principal axes on the trimmed cloud.
    mu, eigvals, eigvecs = _principal_axes(X)
    if eigvecs[1, 1] < 0:
        eigvecs[:, 1] *= -1
    u2_label = mu + eigvecs[:, 1] * np.sqrt(eigvals[1]) * scale * label_offsets[1]
    X = X[np.arange(len(X)) != np.argmin(np.linalg.norm(X - u2_label, axis=1))]

    mu, eigvals, eigvecs = _principal_axes(X)

    # Orient u1 to point right and u2 to point up so the labels sit cleanly.
    if eigvecs[0, 0] < 0:
        eigvecs[:, 0] *= -1
    if eigvecs[1, 1] < 0:
        eigvecs[:, 1] *= -1

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    clean_axes(ax)

    ax.scatter(
        X[:, 0],
        X[:, 1],
        s=MARK.point,
        facecolor=POINT.blue,
        edgecolor=EDGE.blue,
        linewidth=LW.edge,
        zorder=2,
    )

    # A dashed covariance ellipse whose semi-axes match the principal directions
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ax.add_patch(
        Ellipse(
            xy=mu,
            width=2 * scale * np.sqrt(eigvals[0]),
            height=2 * scale * np.sqrt(eigvals[1]),
            angle=angle,
            facecolor="none",
            edgecolor=AXIS,
            linestyle=(0, (4, 3)),
            linewidth=LW.guide,
            zorder=1,
        )
    )

    axis_colors = [EDGE.coral, EDGE.green]
    axis_labels = [r"$u_1$", r"$u_2$"]
    for i in range(2):
        vec = eigvecs[:, i] * np.sqrt(eigvals[i]) * scale
        ax.annotate(
            "",
            xy=mu + vec,
            xytext=mu,
            arrowprops=dict(
                arrowstyle="-|>",
                color=axis_colors[i],
                linewidth=LW.arrow,
                mutation_scale=20,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=4,
        )
        ax.text(
            *(mu + vec * label_offsets[i]),
            axis_labels[i],
            color=axis_colors[i],
            fontsize=FONT.emphasis,
            ha="center",
            va="center",
            zorder=5,
        )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.0, 4.4)
    save(fig, "pca-principal-components.png")


# ---------------------------------------------------------------------------
# 2. Projection of a single point onto the unit vector u
# ---------------------------------------------------------------------------
def figure_projection():
    """proj_u(x) = (x^T u) u: the foot of the perpendicular from x onto span(u)."""
    u_angle = 27.0
    u = np.array([np.cos(np.radians(u_angle)), np.sin(np.radians(u_angle))])
    x = np.array([2.4, 2.7])
    p = (x @ u) * u  # projection of x onto the line through the origin along u

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    clean_axes(ax)

    # The line spanned by u
    t = np.linspace(-0.7, 3.9, 2)
    line = np.outer(t, u)
    ax.plot(line[:, 0], line[:, 1], color=AXIS, linewidth=LW.line, zorder=1)

    # x and the unit vector u as arrows from the origin
    ax.annotate(
        "",
        xy=x,
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>", color=EDGE.coral, linewidth=LW.arrow, mutation_scale=18
        ),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=u,
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>", color=EDGE.green, linewidth=LW.arrow, mutation_scale=18
        ),
        zorder=4,
    )

    # Dashed perpendicular from x down to its projection
    ax.plot(
        [x[0], p[0]],
        [x[1], p[1]],
        color=INK,
        linestyle=(0, (4, 3)),
        linewidth=LW.guide,
        zorder=3,
    )

    # Right-angle marker at the foot of the perpendicular
    res_unit = (x - p) / np.linalg.norm(x - p)
    d = 0.2
    corner = np.array([p - d * u, p - d * u + d * res_unit, p + d * res_unit])
    ax.plot(corner[:, 0], corner[:, 1], color=INK, linewidth=LW.guide, zorder=4)

    ax.scatter(
        *p,
        s=MARK.large,
        facecolor=POINT.blue,
        edgecolor=EDGE.blue,
        linewidth=LW.edge,
        zorder=5,
    )
    ax.scatter(
        *x,
        s=MARK.large,
        facecolor=POINT.coral,
        edgecolor=EDGE.coral,
        linewidth=LW.edge,
        zorder=5,
    )

    ax.text(x[0] - 1.4, x[1] - 1.25, r"$x$", color=EDGE.coral, fontsize=FONT.emphasis)
    ax.text(u[0] - 0.46, u[1] - 0.48, r"$u$", color=EDGE.green, fontsize=FONT.emphasis)
    ax.text(
        p[0] - 0.12,
        p[1] - 0.30,
        r"$\mathrm{proj}_u(x)$",
        color=EDGE.blue,
        fontsize=FONT.title,
        ha="left",
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-0.7, 3.9)
    ax.set_ylim(-0.5, 3.2)
    save(fig, "pca-projection.png")


# ---------------------------------------------------------------------------
# 3. Variance of the projections: a good direction vs. a poor one
# ---------------------------------------------------------------------------
def figure_variance_direction():
    """The same cloud projected onto its max-variance axis (spread out) versus
    its min-variance axis (bunched up)."""
    rng = np.random.default_rng(4)
    cov = np.array([[1.6, 1.05], [1.05, 0.9]])
    X = rng.multivariate_normal([0.0, 0.0], cov, size=90)
    mu, _, eigvecs = _principal_axes(X)
    Xc = X - mu

    panels = [(eigvecs[:, 0], "high variance"), (eigvecs[:, 1], "low variance")]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9))
    for ax, (u, title) in zip(axes, panels):
        if u[1] < 0:
            u = -u
        clean_axes(ax)
        ax.set_aspect("equal")
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)

        # Line spanned by the candidate direction; overshoot the frame so it is
        # clipped to the axes and always extends past every projection
        t = np.array([-8.0, 8.0])
        line = mu + np.outer(t, u)
        ax.plot(line[:, 0], line[:, 1], color=AXIS, linewidth=LW.line, zorder=1)

        # Each point and its projection onto the line
        coords = Xc @ u
        proj = mu + np.outer(coords, u)
        for xi, pi in zip(X, proj):
            ax.plot(
                [xi[0], pi[0]],
                [xi[1], pi[1]],
                color=AXIS,
                linestyle=(0, (3, 3)),
                linewidth=LW.edge,
                zorder=2,
            )
        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=MARK.small,
            facecolor=POINT.blue,
            edgecolor=EDGE.blue,
            linewidth=LW.edge,
            zorder=3,
        )
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            s=MARK.small,
            facecolor=POINT.coral,
            edgecolor=EDGE.coral,
            linewidth=LW.edge,
            zorder=4,
        )
        ax.set_title(title, fontsize=FONT.title, color=INK)

    fig.subplots_adjust(wspace=0.08)
    save(fig, "pca-variance-direction.png")


def main():
    figure_principal_components()
    figure_projection()
    figure_variance_direction()


if __name__ == "__main__":
    main()
