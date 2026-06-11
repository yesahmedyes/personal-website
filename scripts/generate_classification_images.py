import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (
    setup, clean_axes, save,
    INK, AXIS, BACKGROUND,
    HUE, POINT, EDGE, REGION,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# 1. Sigmoid / logistic curve
# ---------------------------------------------------------------------------
def figure_sigmoid():
    z = np.linspace(-8, 8, 400)
    g = 1.0 / (1.0 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    clean_axes(ax)

    # Asymptotes at 0 and 1
    for level in (0.0, 1.0):
        ax.axhline(
            level,
            color=AXIS,
            linestyle=(0, (4, 3)),
            linewidth=LW.guide,
            zorder=1,
        )

    # Vertical reference at z = 0
    ax.axvline(
        0,
        color=AXIS,
        linestyle=(0, (4, 3)),
        linewidth=LW.guide,
        zorder=1,
    )

    # The sigmoid itself
    ax.plot(
        z,
        g,
        color=EDGE.coral,
        linewidth=LW.line,
        zorder=3,
        label=r"$g(z) = \dfrac{1}{1 + e^{-z}}$",
    )

    # Midpoint g(0) = 0.5
    ax.scatter(
        [0],
        [0.5],
        s=MARK.large,
        facecolor=POINT.blue,
        edgecolor=EDGE.blue,
        linewidth=LW.edge,
        zorder=4,
    )
    ax.annotate(
        r"$g(0) = \frac{1}{2}$",
        xy=(0, 0.5),
        xytext=(1.4, 0.36),
        fontsize=FONT.annotation,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=LW.guide),
    )

    ax.text(-7.7, 0.04, r"$0$", fontsize=FONT.annotation, color=INK, va="center")
    ax.text(-7.7, 0.96, r"$1$", fontsize=FONT.annotation, color=INK, va="center")

    ax.set_xlabel(r"$z = 0$")
    ax.set_ylabel(r"$h_\theta(x)$")
    ax.set_ylim(-0.08, 1.12)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.05),
        fontsize=FONT.title,
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
    clean_axes(ax)

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
        colors=[REGION.coral],
        zorder=0,
    )
    ax.contourf(
        gx,
        gy,
        score,
        levels=[score.min(), 0],
        colors=[REGION.blue],
        zorder=0,
    )

    # Decision boundary line
    xs = np.linspace(-lim, lim, 100)
    ys = -(t0 + t1 * xs) / t2
    ax.plot(
        xs,
        ys,
        color=INK,
        linewidth=LW.line,
        zorder=2,
    )

    # Data points
    ax.scatter(
        c0[:, 0],
        c0[:, 1],
        s=MARK.point,
        facecolor=POINT.blue,
        edgecolor=EDGE.blue,
        linewidth=LW.edge,
        zorder=3,
        label=r"$y^{(i)} = 0$",
    )
    ax.scatter(
        c1[:, 0],
        c1[:, 1],
        s=MARK.point,
        facecolor=POINT.coral,
        edgecolor=EDGE.coral,
        linewidth=LW.edge,
        zorder=3,
        label=r"$y^{(i)} = 1$",
    )

    # Region annotations
    ax.text(
        2.7,
        3.9,
        r"$h_\theta(x) > \frac{1}{2}$",
        fontsize=FONT.annotation,
        color=EDGE.coral,
        ha="center",
    )
    ax.text(
        -3.0,
        -3.9,
        r"$h_\theta(x) < \frac{1}{2}$",
        fontsize=FONT.annotation,
        color=EDGE.blue,
        ha="center",
    )

    # Label the boundary, sitting just off the line and rotated to match it
    boundary_angle = np.degrees(np.arctan2(-(t1 / t2), 1.0))
    ax.text(
        xs[70] + 0.45,
        ys[70] + 0.55,
        r"$\theta^T x = 0$",
        fontsize=FONT.annotation,
        color=INK,
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
        fontsize=FONT.annotation,
        handlelength=1.0,
        borderaxespad=0.6,
        framealpha=1.0,
    )
    legend.get_frame().set_facecolor(BACKGROUND)
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(5)
    save(fig, "classification-decision-boundary.png")


# ---------------------------------------------------------------------------
# 3. Multiclass softmax decision regions
# ---------------------------------------------------------------------------
def figure_softmax_regions():
    """Three-class softmax partition: predicted class = argmax_i theta_i^T x."""
    rng = np.random.default_rng(7)
    n = 16
    lim = 5.0

    # Three class "directions" 120 degrees apart give a symmetric partition.
    # Each class score is theta_i^T x; the prediction is argmax_i theta_i^T x.
    centre_angles = np.array([90.0, 210.0, 330.0])  # cluster centres (degrees)
    centres = 2.6 * np.column_stack(
        [np.cos(np.radians(centre_angles)), np.sin(np.radians(centre_angles))]
    )
    thetas = centres  # theta_i points toward cluster i

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    clean_axes(ax)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Predicted class over a grid = argmax of the linear scores
    gx, gy = np.meshgrid(
        np.linspace(-lim, lim, 500),
        np.linspace(-lim, lim, 500),
    )
    scores = np.stack([thetas[i, 0] * gx + thetas[i, 1] * gy for i in range(3)])
    cls = np.argmax(scores, axis=0).astype(float)

    region_colors = [
        REGION.blue,
        REGION.coral,
        REGION.sage,
    ]
    ax.contourf(
        gx,
        gy,
        cls,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=region_colors,
        zorder=0,
    )

    # Decision boundaries: rays from the origin where two scores tie,
    # bisecting adjacent cluster directions (at 30, 150, 270 degrees).
    # Extend past the axes so each ray is clipped flush to the plot edge.
    ray = 2 * lim
    for a in (30.0, 150.0, 270.0):
        ax.plot(
            [0, ray * np.cos(np.radians(a))],
            [0, ray * np.sin(np.radians(a))],
            color=INK,
            linewidth=LW.line,
            zorder=2,
        )

    point_fill = [
        POINT.blue,
        POINT.coral,
        POINT.green,
    ]
    point_edge = [
        EDGE.blue,
        EDGE.coral,
        EDGE.green,
    ]
    for i in range(3):
        pts = rng.normal(loc=centres[i], scale=0.62, size=(n, 2))
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=MARK.point,
            facecolor=point_fill[i],
            edgecolor=point_edge[i],
            linewidth=LW.edge,
            zorder=3,
            label=rf"$y^{{(i)}} = {i + 1}$",
        )

    # Label a boundary with the "two classes tie" condition
    ax.text(
        0.3,
        -4.0,
        r"$\theta_i^T x = \theta_j^T x$",
        fontsize=FONT.annotation,
        color=INK,
        ha="left",
        va="center",
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    legend = ax.legend(
        loc="upper right",
        fontsize=FONT.annotation,
        handlelength=1.0,
        borderaxespad=0.6,
        facecolor=BACKGROUND,
        edgecolor="none",
        framealpha=1.0,
    )
    legend.set_zorder(5)
    save(fig, "classification-softmax-regions.png")


# ---------------------------------------------------------------------------
# 4. Logistic (cross-entropy) loss vs. the score t = theta^T x
# ---------------------------------------------------------------------------
def figure_logistic_loss():
    """How the per-example logistic loss penalises the score for each label."""
    t = np.linspace(-6, 6, 400)
    loss_y1 = np.log1p(np.exp(-t))  # y = 1 : -log h_theta(x)
    loss_y0 = np.log1p(np.exp(t))  # y = 0 : -log(1 - h_theta(x))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    clean_axes(ax)

    # The loss floor (both curves approach 0) and the t = 0 reference
    ax.axhline(
        0.0,
        color=AXIS,
        linestyle=(0, (4, 3)),
        linewidth=LW.guide,
        zorder=1,
    )
    ax.axvline(
        0.0,
        color=AXIS,
        linestyle=(0, (4, 3)),
        linewidth=LW.guide,
        zorder=1,
    )

    ax.plot(
        t,
        loss_y1,
        color=EDGE.coral,
        linewidth=LW.line,
        zorder=3,
        label=r"$y = 1$",
    )
    ax.plot(
        t,
        loss_y0,
        color=EDGE.blue,
        linewidth=LW.line,
        zorder=3,
        label=r"$y = 0$",
    )

    # Intuition: confidently wrong is punished hard, confidently right costs ~0
    ax.text(
        -4.8,
        5.3,
        "confidently\nwrong",
        fontsize=FONT.tick,
        color=INK,
        ha="left",
        va="center",
        ma="center",
    )
    ax.text(
        3.3,
        0.6,
        "confidently\nright",
        fontsize=FONT.tick,
        color=INK,
        ha="left",
        va="center",
        ma="center",
    )
    # Mirror labels on the y = 0 (blue) curve: low loss (right) on the left,
    # high loss (wrong) on the right
    ax.text(
        -3.3,
        0.6,
        "confidently\nright",
        fontsize=FONT.tick,
        color=INK,
        ha="right",
        va="center",
        ma="center",
    )
    ax.text(
        4.8,
        5.3,
        "confidently\nwrong",
        fontsize=FONT.tick,
        color=INK,
        ha="right",
        va="center",
        ma="center",
    )

    ax.set_xlabel(r"$t = 0$")
    ax.set_ylabel(r"$\ell_{\mathrm{logistic}}(t, y)$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.3, 6.2)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=FONT.annotation,
        handlelength=1.6,
        borderaxespad=0.6,
        borderpad=0.8,
        framealpha=1.0,
    )
    legend.get_frame().set_facecolor(BACKGROUND)
    legend.get_frame().set_edgecolor(AXIS)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(5)
    save(fig, "classification-logistic-loss.png")


def main():
    figure_sigmoid()
    figure_decision_boundary()
    figure_softmax_regions()
    figure_logistic_loss()


if __name__ == "__main__":
    main()
