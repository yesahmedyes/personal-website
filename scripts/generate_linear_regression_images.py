import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from style import (
    setup, clean_axes, save,
    INK, AXIS, HUE, POINT, EDGE, TINT,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# 1. Scatter + fitted line + residuals
# ---------------------------------------------------------------------------
def figure_residuals():
    rng = np.random.default_rng(7)
    n = 12
    x = np.linspace(1.0, 9.0, n)
    true_slope, true_intercept = 0.85, 1.2
    y = true_intercept + true_slope * x + rng.normal(0, 1.1, size=n)

    # Ordinary least squares fit
    X = np.vstack([np.ones_like(x), x]).T
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ theta

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    clean_axes(ax, spine=INK)

    # Residual segments (dashed, drawn under the points)
    for xi, yi, yh in zip(x, y, y_hat):
        ax.plot(
            [xi, xi],
            [yi, yh],
            color=AXIS,
            linestyle=(0, (4, 3)),
            linewidth=LW.guide,
            zorder=1,
        )

    # Fitted line
    xs = np.linspace(x.min() - 0.6, x.max() + 0.6, 100)
    ax.plot(
        xs,
        theta[0] + theta[1] * xs,
        color=EDGE.coral,
        linewidth=LW.line,
        zorder=2,
        label=r"$h_\theta(x) = \theta^T x$",
    )

    # Data points
    ax.scatter(
        x,
        y,
        s=MARK.large,
        facecolor=POINT.blue,
        edgecolor=EDGE.blue,
        linewidth=LW.edge,
        zorder=3,
        label=r"$(x^{(i)},\, y^{(i)})$",
    )

    # Highlight one residual with an annotation
    idx = 3
    mid_y = (y[idx] + y_hat[idx] + 0.1) / 2
    ax.annotate(
        "",
        xy=(x[idx], mid_y),
        xytext=(x[idx] + 1.5, mid_y - 1.0),
        arrowprops=dict(
            arrowstyle="-",
            color=INK,
            linewidth=LW.guide,
        ),
    )
    ax.text(
        x[idx] + 1.58,
        mid_y - 1.15,
        r"$h_\theta(x^{(i)}) - y^{(i)}$",
        color=INK,
        fontsize=FONT.annotation,
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=FONT.annotation,
        handlelength=1.6,
        borderaxespad=0.2,
    )
    save(fig, "linear-regression-residuals.png")


# ---------------------------------------------------------------------------
# 2. Cost function contours + gradient descent path
# ---------------------------------------------------------------------------
def figure_cost():
    # A tiny 1-feature dataset so J depends on (theta0, theta1).
    # Center x (so the two axes are uncorrelated) but give it a non-unit
    # spread: this makes the cost contours elliptical rather than circular,
    # so gradient descent follows a curved path down into the valley.
    rng = np.random.default_rng(3)
    x = np.linspace(0, 5, 20)
    x = (x - x.mean()) / x.std() * 2.0
    y = 1.0 + 1.5 * x + rng.normal(0, 1.0, size=x.size)
    X = np.vstack([np.ones_like(x), x]).T

    def cost(t0, t1):
        pred = t0 + t1 * x
        return 0.5 * np.sum((pred - y) ** 2)

    # Optimal theta (center of the bowl)
    theta_opt = np.linalg.lstsq(X, y, rcond=None)[0]

    span = 4.5
    t0_range = np.linspace(theta_opt[0] - span, theta_opt[0] + span, 240)
    t1_range = np.linspace(theta_opt[1] - span, theta_opt[1] + span, 240)
    T0, T1 = np.meshgrid(t0_range, t1_range)
    J = 0.5 * ((T0[..., None] + T1[..., None] * x - y) ** 2).sum(axis=-1)

    # Gradient descent path. With elliptical contours the steep theta_1
    # direction is corrected quickly while theta_0 drifts slowly, producing a
    # curved "elbow" trajectory into the minimum.
    theta = np.array([theta_opt[0] - 4.0, theta_opt[1] + 3.4])
    alpha = 0.01
    path = [theta.copy()]
    for _ in range(45):
        grad = X.T @ (X @ theta - y)
        theta = theta - alpha * grad
        path.append(theta.copy())
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    clean_axes(ax, spine=INK)

    j_min = cost(*theta_opt)
    j_start = cost(*path[0])
    levels = j_min + np.linspace(0.04, 1.0, 9) ** 2 * (j_start - j_min)
    ax.contour(
        T0,
        T1,
        J,
        levels=levels,
        colors=TINT.blue,
        linewidths=LW.guide,
    )

    # Descent path
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=INK,
        linewidth=LW.line,
        zorder=2,
    )
    ax.scatter(
        path[1:, 0],
        path[1:, 1],
        s=MARK.small,
        facecolor=POINT.coral,
        edgecolor=EDGE.coral,
        linewidth=LW.edge,
        zorder=3,
    )

    # Start and minimum markers
    ax.scatter(*path[0], s=MARK.large, color=EDGE.coral, zorder=4)
    ax.annotate(
        "start",
        xy=path[0],
        xytext=(path[0, 0], path[0, 1] + 0.35),
        fontsize=FONT.annotation,
        color=INK,
        ha="center",
    )
    ax.scatter(
        *theta_opt,
        marker="*",
        s=MARK.star,
        color=HUE.green,
        edgecolor=EDGE.green,
        linewidth=LW.guide,
        zorder=4,
    )
    ax.annotate(
        r"$\min_\theta\, J(\theta)$",
        xy=theta_opt,
        xytext=(theta_opt[0] + 0.5, theta_opt[1] - 1.4),
        fontsize=FONT.annotation,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, linewidth=LW.guide),
    )

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    save(fig, "linear-regression-cost.png")


# ---------------------------------------------------------------------------
# 3. Orthogonal projection of y onto the column space of X
# ---------------------------------------------------------------------------
def figure_projection():
    # Two columns of X span a plane in R^3; y is projected onto that plane.
    c1 = np.array([1.0, 0.2, 0.25])
    c2 = np.array([0.25, 1.0, 0.15])
    c1 /= np.linalg.norm(c1)
    c2 /= np.linalg.norm(c2)

    y = np.array([0.9, 0.8, 1.7])

    # Projection of y onto span(c1, c2)
    A = np.vstack([c1, c2]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = A @ coef

    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_position([0, 0, 1, 1])

    # Build the plane patch from the two spanning columns
    s = np.linspace(-1.3, 1.9, 2)
    t = np.linspace(-1.3, 1.9, 2)
    S, T = np.meshgrid(s, t)
    plane = S[..., None] * c1[None, None, :] + T[..., None] * c2[None, None, :]
    ax.plot_surface(
        plane[..., 0],
        plane[..., 1],
        plane[..., 2],
        color=TINT.blue,
        alpha=0.45,  # noqa: style -- a 3D plane needs real translucency for depth
        linewidth=0,
        shade=False,
        zorder=1,
    )

    def arrow(vec, color, lw=LW.arrow, label=None, offset=None):
        ax.quiver(
            0,
            0,
            0,
            vec[0],
            vec[1],
            vec[2],
            color=color,
            linewidth=lw,
            arrow_length_ratio=0.12,
        )
        if label is not None:
            pos = vec * 1.05 if offset is None else vec + offset
            ax.text(*pos, label, color=color, fontsize=FONT.title)

    # y, projection y_hat, and the perpendicular residual
    arrow(y, EDGE.coral, label=r"$\vec{y}$")
    arrow(
        y_hat,
        EDGE.blue,
        label=r"$\hat{y} = X\theta$",
        offset=np.array([-0.15, -0.05, -0.42]),
    )

    ax.plot(
        [y_hat[0], y[0]],
        [y_hat[1], y[1]],
        [y_hat[2], y[2]],
        color=INK,
        linestyle=(0, (4, 3)),
        linewidth=LW.line,
        zorder=5,
    )
    res_mid = (y + y_hat) / 2
    ax.text(
        res_mid[0] + 0.08,
        res_mid[1] + 0.08,
        res_mid[2],
        r"$\vec{y} - \hat{y}$",
        color=INK,
        fontsize=FONT.annotation,
    )

    # Small right-angle marker at the foot of the projection
    n = y - y_hat
    n_unit = n / np.linalg.norm(n)
    in_plane = c1 / np.linalg.norm(c1)
    p0 = y_hat
    d = 0.18
    corner = [
        p0 + d * in_plane,
        p0 + d * in_plane + d * n_unit,
        p0 + d * n_unit,
    ]
    corner = np.array(corner)
    ax.plot(
        corner[:, 0],
        corner[:, 1],
        corner[:, 2],
        color=INK,
        linewidth=LW.guide,
        zorder=6,
    )

    # Label the plane near a clear corner, lifted slightly off the surface
    # along the plane normal so the translucent fill doesn't cover it.
    normal = np.cross(c1, c2)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    label_pt = -0.5 * c1 - 0.1 * c2 + 0.2 * normal

    ax.text(
        label_pt[0],
        label_pt[1],
        label_pt[2],
        r"$\mathrm{col}(X)$",
        color=INK,
        fontsize=FONT.title,
        zorder=10,
        ha="right",
    )

    # Cosmetic clean-up of the 3d axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color((1, 1, 1, 0))
    ax.view_init(elev=18, azim=-58)

    save(fig, "linear-regression-projection.png", pad_inches=0.02, trim=True)


def main():
    figure_residuals()
    figure_cost()
    figure_projection()


if __name__ == "__main__":
    main()
