import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgb, to_rgba

# Output directory (repo_root/public/images/notes), resolved relative to this file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(REPO_ROOT, "public", "images", "notes")

DPI = 200


# ---------------------------------------------------------------------------
# Compositing: bake a translucent fill onto the white page once, here.
# ---------------------------------------------------------------------------
def _over_white(hex_color, alpha):
    """Composite ``hex_color`` at ``alpha`` over white -> a flat opaque (r,g,b)."""
    r, g, b = to_rgb(hex_color)
    return (r * alpha + (1 - alpha), g * alpha + (1 - alpha), b * alpha + (1 - alpha))


# --- base hues (private; call sites use the role tokens below) -------------
_BLUE, _CORAL, _GREEN = "#81B1D5", "#FD8D86", "#8BD48A"          # point fills
_BLUE_D, _CORAL_D, _GREEN_D = "#639DCA", "#FC7269", "#6EC96E"     # dark edges
_L_BLUE, _L_CORAL, _L_SAGE = "#BFD8EA", "#FFC5C0", "#C5EAC4"      # light fills
_INK, _AXIS, _BG = "#3F3F3F", "#747474", "#FFFFFF"

# --- the only opacities in the whole figure set ----------------------------
_A_POINT = 0.85      # scatter data points
_A_FADED = 0.5       # faded / unassigned (grey) points
_A_REGION = 0.40     # decision / cluster zone fills (light hue, pre-composited)
_A_BAND = 0.20       # Gaussian density bands (outer light / inner point hue)


# --- neutrals --------------------------------------------------------------
INK = _INK           # all text, arrows, decision boundaries, glyphs
AXIS = _AXIS         # spines, grid, faded outlines
BACKGROUND = _BG


class HUE:
    """Opaque base point colours -- ellipse borders, solid marks, line labels."""
    blue = _BLUE
    coral = _CORAL
    green = _GREEN


class POINT:
    """Scatter faces -- the canonical point opacity is baked in (never pass alpha=)."""
    blue = to_rgba(_BLUE, _A_POINT)
    coral = to_rgba(_CORAL, _A_POINT)
    green = to_rgba(_GREEN, _A_POINT)
    muted = to_rgba(_AXIS, _A_FADED)     # faded / unassigned grey points


class EDGE:
    """Opaque point / marker outlines."""
    blue = _BLUE_D
    coral = _CORAL_D
    green = _GREEN_D
    neutral = _AXIS


class REGION:
    """Flat decision / cluster zone fills -- never drawn with alpha=."""
    blue = _over_white(_L_BLUE, _A_REGION)
    coral = _over_white(_L_CORAL, _A_REGION)
    sage = _over_white(_L_SAGE, _A_REGION)


class BAND:
    """Flat Gaussian density bands: outer = light hue, inner = point hue."""
    outer_blue = _over_white(_L_BLUE, _A_BAND)
    inner_blue = _over_white(_BLUE, _A_BAND)
    outer_coral = _over_white(_L_CORAL, _A_BAND)
    inner_coral = _over_white(_CORAL, _A_BAND)
    outer_green = _over_white(_L_SAGE, _A_BAND)
    inner_green = _over_white(_GREEN, _A_BAND)


class TINT:
    """Opaque pastel fills for SOLID diagram shapes (boxes, funnels, bars).

    Unlike REGION (a translucent overlay composited onto white), these are the
    light hues at full strength -- use them where the original figure filled a
    shape with a flat ``light_*`` colour and no alpha.
    """
    blue = _L_BLUE
    coral = _L_CORAL
    sage = _L_SAGE


# Index sequences for multiclass loops (replace per-script FILL/EDGE/LIGHT lists).
HUES = [HUE.blue, HUE.coral, HUE.green]
POINTS = [POINT.blue, POINT.coral, POINT.green]
EDGES = [EDGE.blue, EDGE.coral, EDGE.green]
REGIONS = [REGION.blue, REGION.coral, REGION.sage]
BANDS_OUTER = [BAND.outer_blue, BAND.outer_coral, BAND.outer_green]
BANDS_INNER = [BAND.inner_blue, BAND.inner_coral, BAND.inner_green]


# ---------------------------------------------------------------------------
# Scales: one named tier per semantic role.
# ---------------------------------------------------------------------------
class FONT:
    tick = 11           # axis number labels
    annotation = 12     # in-axes math / value labels (also the rc base)
    title = 13          # panel & axis titles, axis labels
    emphasis = 15       # hero labels (class labels, big arrow labels)
    glyph = 24          # giant equation operators (+, =)


class LW:
    edge = 0.7          # scatter / marker outlines
    guide = 1.2         # references, grids, dashed guides, ellipse borders
    line = 1.8          # main plotted curves, densities, decision boundaries
    arrow = 2.6         # bold directional arrows (principal axes, mixing)


class MARK:
    small = 30          # dense / small scatter
    point = 42          # standard data scatter
    large = 70          # emphasized data points
    star = 300          # cluster centroids / minima markers


# ---------------------------------------------------------------------------
# Global setup -- replaces the rcParams block copy-pasted into every script.
# ---------------------------------------------------------------------------
def setup(axes_edge=AXIS):
    """Apply the shared rcParams. Call once at import time in each script.

    ``axes_edge`` overrides the spine colour (linear-regression's 3D panels use
    INK for readability); everything else stays fixed.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],   # pin for cross-machine reproducibility
            "font.size": FONT.annotation,
            "axes.titlesize": FONT.title,
            "axes.labelsize": FONT.title,
            "xtick.labelsize": FONT.tick,
            "ytick.labelsize": FONT.tick,
            "legend.fontsize": FONT.annotation,
            "lines.linewidth": LW.line,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": axes_edge,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
        }
    )


# ---------------------------------------------------------------------------
# Shared drawing helpers (previously duplicated in every script).
# ---------------------------------------------------------------------------
def clean_axes(ax, spine=AXIS):
    """Light, minimal 2-D axes: drop top/right spines, no ticks.

    ``spine`` is the colour of the left/bottom spines (default AXIS; linear
    regression's plot panels pass INK for darker, more readable axes).
    """
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(spine)
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_yticks([])


def clean_axes_off(ax):
    """Fully bare axes (the em-algorithm / diagram idiom)."""
    ax.set_axis_off()


def cov_ellipse(ax, mean, cov, nstd, *, border, fill="none", lw=LW.guide,
                ls="solid", z=3):
    """A constant-density contour of ``N(mean, cov)`` at ``nstd`` std devs.

    ``border`` is an opaque colour token; ``fill`` is a flat colour token (e.g.
    ``BAND.outer_blue``) or ``"none"``. The border stays crisp; the fill carries
    its opacity through pre-compositing, so no per-call alpha is ever needed.
    """
    import numpy as np

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nstd * np.sqrt(vals)
    linestyle = (0, (4, 3)) if ls == "dashed" else "solid"
    ax.add_patch(Ellipse(mean, width, height, angle=angle, facecolor=fill,
                         edgecolor=border, linewidth=lw, linestyle=linestyle,
                         zorder=z))


def point_rgba(rgb):
    """Attach the canonical scatter opacity to an RGB colour or (..., 3) array.

    For data points whose face colour is *computed* (e.g. a per-point gradient
    coloured by some weight), so they carry the same opacity as the ``POINT.*``
    tokens without a script ever writing a raw ``alpha=``.
    """
    import numpy as np

    arr = np.asarray(rgb, dtype=float)
    alpha = np.full(arr.shape[:-1] + (1,), _A_POINT)
    return np.concatenate([arr, alpha], axis=-1)


def save(fig, name, *, pad_inches=0.25, trim=False):
    """Write ``fig`` to ``OUT_DIR/name``. ``trim`` crops surrounding whitespace."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    if trim:
        _trim_whitespace(path)
    print(f"wrote {path}")


def _trim_whitespace(path, padding=16, threshold=250):
    """Crop white margins from a saved PNG, leaving a small padding (3-D figures)."""
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image)
    content = (pixels[:, :, 3] > 0) & np.any(pixels[:, :, :3] < threshold, axis=2)
    coords = np.argwhere(content)
    if coords.size == 0:
        return
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    x0, y0 = max(x0 - padding, 0), max(y0 - padding, 0)
    x1, y1 = min(x1 + padding, image.width), min(y1 + padding, image.height)
    image.crop((x0, y0, x1, y1)).save(path)
