import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import (
    setup, clean_axes, save,
    INK, HUE, EDGE, TINT,
    FONT, LW, MARK,
)

setup()


# ---------------------------------------------------------------------------
# 1. Agent-environment interaction loop
# ---------------------------------------------------------------------------
def figure_agent_environment():
    """The canonical RL cycle: the agent takes an action; the environment
    returns a new state and reward."""
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-1.8, 1.8)

    b = 1.25  # vertical offset of each box from the centre

    def box(cx, cy, w, h, label, face, edge):
        ax.add_patch(
            Rectangle(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                facecolor=face,
                edgecolor=edge,
                linewidth=LW.guide,
                zorder=3,
            )
        )
        ax.text(cx, cy, label, ha="center", va="center", fontsize=FONT.emphasis, color=INK, zorder=4)

    w_agent, w_env = 1.7, 3.0
    box(0, b, w_agent, 0.8, "Agent", TINT.blue, EDGE.blue)
    box(0, -b, w_env, 0.8, "Environment", TINT.coral, EDGE.coral)

    # Two curved arrows form the loop. Each is a single patch (so no doubled
    # line) that lands at the middle of a box side; shrink=0 puts the tip exactly
    # on the box edge.
    arc = dict(
        arrowstyle="-|>",
        color=INK,
        linewidth=LW.line,
        mutation_scale=18,
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=-0.8",
    )
    # Action: down the right, Agent -> Environment
    ax.annotate("", xy=(w_env / 2, -b), xytext=(w_agent / 2, b), arrowprops=arc, zorder=1)
    # State + reward: up the left, Environment -> Agent
    ax.annotate("", xy=(-w_agent / 2, b), xytext=(-w_env / 2, -b), arrowprops=arc, zorder=1)

    # Labels outside the loop
    ax.text(-3.1, 0.0, "State $s_t$\nReward $r_t$", ha="center", va="center", fontsize=FONT.emphasis, color=INK, linespacing=1.9)
    ax.text(3.1, 0.0, "Action $a_t$", ha="center", va="center", fontsize=FONT.emphasis, color=INK)

    save(fig, "rl-agent-environment.png")


# ---------------------------------------------------------------------------
# 2. Sampled rollouts through a continuous state space
# ---------------------------------------------------------------------------
def figure_rollouts():
    """Several simulated trajectories s_0 -> s_1 -> ... -> s_T used to fit a model."""
    rng = np.random.default_rng(5)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    clean_axes(ax)

    # Well-separated starts with a shared drift keep the three rollouts parallel
    # and visually distinct instead of tangling together.
    specs = [
        (np.array([-2.9, -0.4]), (HUE.blue, EDGE.blue)),
        (np.array([-2.7, 1.2]), (HUE.coral, EDGE.coral)),
        (np.array([-2.7, -2.0]), (HUE.green, EDGE.green)),
    ]
    drift = np.array([0.85, 0.42])
    T = 7

    all_pts = []
    for k, (s0, (fill, edge)) in enumerate(specs):
        pts = [s0.astype(float)]
        for _ in range(T):
            pts.append(pts[-1] + drift + rng.normal(0, 0.3, size=2))
        pts = np.array(pts)
        all_pts.append(pts)

        for i in range(len(pts) - 1):
            ax.annotate(
                "",
                xy=pts[i + 1],
                xytext=pts[i],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=edge,
                    linewidth=LW.line,
                    mutation_scale=11,
                    shrinkA=7,
                    shrinkB=7,
                ),
            )
        ax.scatter(pts[1:, 0], pts[1:, 1], s=MARK.point, facecolor=fill, edgecolor=edge, linewidth=LW.edge, zorder=3)
        # Start state, marked and labelled
        ax.scatter(*pts[0], s=MARK.star, marker="*", facecolor=fill, edgecolor=edge, linewidth=LW.guide, zorder=4)
        ax.text(pts[0, 0] - 0.1, pts[0, 1] - 0.5, rf"$s_0^{{({k + 1})}}$", ha="center", fontsize=FONT.annotation, color=edge)

    # Label one transition on the first (blue) trajectory in open space
    p0, p1 = all_pts[0][3], all_pts[0][4]
    mid = (p0 + p1) / 2
    ax.text(mid[0] + 0.05, mid[1] - 0.42, r"$a_t^{(i)}$", ha="center", fontsize=FONT.annotation, color=EDGE.blue)

    pts = np.vstack(all_pts)
    ax.set_xlim(pts[:, 0].min() - 0.7, pts[:, 0].max() + 0.7)
    ax.set_ylim(pts[:, 1].min() - 0.9, pts[:, 1].max() + 0.7)
    ax.set_xlabel(r"$s_{[1]}$")
    ax.set_ylabel(r"$s_{[2]}$")
    save(fig, "rl-rollouts.png")


def main():
    figure_agent_environment()
    figure_rollouts()


if __name__ == "__main__":
    main()
