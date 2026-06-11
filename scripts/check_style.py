import ast
import glob
import os
import re
import sys

import style  # the source of truth; also gives us the valid token names

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Token namespaces and their valid attribute names (catches typos like FONT.titel).
_TOKEN_CLASSES = ["FONT", "LW", "MARK", "POINT", "EDGE", "REGION", "BAND", "HUE"]
_VALID_ATTRS = {
    name: {k for k in vars(getattr(style, name)) if not k.startswith("_")}
    for name in _TOKEN_CLASSES
}

# Keyword args whose numeric value must come from a scale.
_NUM_RULES = {
    "alpha": "FONT/POINT/... token (opacity is baked into the colour token)",
    "fontsize": "a FONT.* tier",
    "linewidth": "an LW.* tier",
    "lw": "an LW.* tier",
    "s": "a MARK.* tier",
    "markersize": "a MARK.* tier",
    "ms": "a MARK.* tier",
}
_LW_KEYS = {"linewidth", "lw"}            # these may be literal 0 ("no line")

# Keyword args whose colour value must come from a colour token.
_COLOR_KEYS = {"color", "facecolor", "edgecolor", "c", "colors", "ec", "fc"}
_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_GRAY_RE = re.compile(r"^(?:0?\.\d+|0|1(?:\.0+)?)$")   # "0.8", ".5", "0", "1"
_NAMED = {
    "black", "white", "red", "green", "blue", "coral", "orange", "purple",
    "gray", "grey", "cyan", "magenta", "yellow",
    "k", "w", "r", "g", "b", "c", "m", "y",
}


def _is_number(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


def _bad_color_literal(node):
    """True if ``node`` is a hard-coded colour (hex / named / grayscale / rgb tuple)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        v = node.value.strip()
        if v.lower() == "none":
            return False
        return bool(_HEX_RE.match(v) or _GRAY_RE.match(v) or v.lower() in _NAMED)
    if isinstance(node, (ast.Tuple, ast.List)):
        elts = node.elts
        return 0 < len(elts) <= 4 and all(_is_number(e) for e in elts)
    return False


def _check_keyword(kw, call_name, lineno, add):
    """Validate one ``name=value`` keyword on a call."""
    name = kw.arg
    if name is None:
        return
    if name in _NUM_RULES and _is_number(kw.value):
        if name in _LW_KEYS and kw.value.value == 0:
            return  # linewidth=0 means "no line", allowed
        add(lineno, f"{name}={kw.value.value!r} -> use {_NUM_RULES[name]}")
    if name in _COLOR_KEYS and _bad_color_literal(kw.value):
        add(lineno, f"hard-coded colour in {name}= -> use a colour token from style")


def check_file(path):
    """Return a list of (lineno, message) violations for one script."""
    with open(path) as f:
        src = f.read()
    src_lines = src.splitlines()
    tree = ast.parse(src, filename=path)
    out = []

    def add(lineno, msg):
        line = src_lines[lineno - 1] if 0 < lineno <= len(src_lines) else ""
        if "noqa" in line:
            return
        out.append((lineno, msg))

    for node in ast.walk(tree):
        # raw literals passed to a drawing call
        if isinstance(node, ast.Call):
            fn = node.func
            call_name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            for kw in node.keywords:
                line = getattr(kw.value, "lineno", node.lineno)
                _check_keyword(kw, call_name, line, add)

        # literals hidden in a helper's default arguments
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            pairs = list(zip(a.args[len(a.args) - len(a.defaults):], a.defaults))
            pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d]
            for arg, default in pairs:
                fake = ast.keyword(arg=arg.arg, value=default)
                _check_keyword(fake, "", default.lineno, add)

        # PALETTE subscripts must become tokens
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id == "PALETTE":
            add(node.lineno, "PALETTE[...] -> use a colour token from style")

        # rcParams must go through style.setup()
        if (isinstance(node, ast.Attribute) and node.attr == "rcParams") or \
                (isinstance(node, ast.Name) and node.id == "rcParams"):
            add(node.lineno, "rcParams set outside style.py -> use style.setup()")

        # importing the old colours module
        if isinstance(node, ast.ImportFrom) and node.module == "colors":
            add(node.lineno, "from colors import ... -> import from style")
        if isinstance(node, ast.Import) and any(n.name == "colors" for n in node.names):
            add(node.lineno, "import colors -> import from style")

        # typo'd token names (FONT.titel, MARK.starr, ...)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in _VALID_ATTRS:
            cls = node.value.id
            if node.attr not in _VALID_ATTRS[cls]:
                add(node.lineno, f"unknown token {cls}.{node.attr}")

    return sorted(out)


def main():
    targets = sorted(glob.glob(os.path.join(SCRIPT_DIR, "generate_*_images.py")))
    extra = os.path.join(SCRIPT_DIR, "_fa_cov_variants.py")
    if os.path.exists(extra):
        targets.append(extra)

    total = 0
    for path in targets:
        violations = check_file(path)
        if violations:
            total += len(violations)
            rel = os.path.relpath(path)
            for lineno, msg in violations:
                print(f"{rel}:{lineno}: {msg}")

    if total:
        print(f"\n{total} style violation(s) across {len(targets)} file(s).")
        return 1
    print(f"OK: {len(targets)} file(s) clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
