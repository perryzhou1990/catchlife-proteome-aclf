# -*- coding: utf-8 -*-
"""Export each panel of a figure script as a standalone high-res PNG.

Usage: python export_panels.py <figdir> <script_path> <prefix>
Output: <figdir>/panels/<prefix>_panel<X>.png  (600 dpi)

Panel grouping is hardcoded per figure (PANEL_GROUPS) because automatic
letter/colourbar detection proved unreliable. Axes indices come from the
probe dumps of each fig*_make.py.
"""
import sys, os, io, runpy
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from PIL import Image

# axes-index -> panel letter, per figure prefix
PANEL_GROUPS = {
    "fig1": {"A": [0], "B": [1], "C": [2]},
    "fig2": {"A": [0, 1], "B": [2], "C": [3, 5], "D": [4, 6]},
    "fig3": {"A": [0], "B": [2], "C": [3], "D": [1]},
    "fig4": {"A": [0], "B": [1], "C": [2], "D": [3], "E": [4]},
    "fig5": {"A": [0], "B": [1], "C": [2, 3]},
    "fig6": {"A": [0], "B": [1], "C": [2], "D": [3, 4]},
    "figS1": {"A": [0], "B": [1], "C": [2, 5], "D": [3], "E": [4]},
}


def to_in(bb, figw_in, fig):
    if bb is None:
        return None
    if bb.x1 > figw_in * 1.5:  # display pixels -> inches
        s = fig.dpi
        bb = Bbox([[bb.x0 / s, bb.y0 / s], [bb.x1 / s, bb.y1 / s]])
    return bb


def export_panels(fig, outdir, prefix, dpi=600, pad=0.07):
    os.makedirs(outdir, exist_ok=True)
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    figw_in, figh_in = fig.get_size_inches()

    groups = PANEL_GROUPS.get(prefix)
    if groups is None:
        raise SystemExit("no PANEL_GROUPS entry for %s" % prefix)

    # bbox (inches) per axes
    bbs = {}
    for i, ax in enumerate(fig.axes):
        try:
            bbs[i] = to_in(ax.get_tightbbox(ren), figw_in, fig)
        except Exception:
            bbs[i] = None

    # union bbox per panel
    panels = {}
    for letter, idxs in groups.items():
        acc = None
        for i in idxs:
            if bbs.get(i) is None:
                continue
            acc = bbs[i] if acc is None else Bbox.union([acc, bbs[i]])
        if acc is not None:
            panels[letter] = acc

    # render the whole figure on a canvas covering all panels.
    # NOTE: fig.get_tightbbox is unreliable here (can return a partial box),
    # so build the canvas from the panel bboxes themselves.
    full = None
    for bb in panels.values():
        full = bb if full is None else Bbox.union([full, bb])
    cpad = 0.08
    canvas_bb = Bbox([[full.x0 - cpad, full.y0 - cpad], [full.x1 + cpad, full.y1 + cpad]])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches=canvas_bb)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    W, H = img.size

    def crop_rect(bb):
        x0 = (bb.x0 - canvas_bb.x0) * dpi
        x1 = (bb.x1 - canvas_bb.x0) * dpi
        y1 = H - (bb.y0 - canvas_bb.y0) * dpi
        y0 = H - (bb.y1 - canvas_bb.y0) * dpi
        return (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))

    paths = []
    for letter, bb in panels.items():
        x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
        # clip only against DISJOINT neighbours (midpoint of the empty gap).
        # Overlapping (contested) tight bboxes are left unclipped: splitting a
        # contested strip would cut real content (titles/labels), so we keep
        # the full tight box and accept a small bleed of the neighbour.
        for m, rb in panels.items():
            if m == letter:
                continue
            y_ov = min(bb.y1, rb.y1) - max(bb.y0, rb.y0) > 0
            x_ov = min(bb.x1, rb.x1) - max(bb.x0, rb.x0) > 0
            if rb.x0 >= bb.x1 and y_ov:          # neighbour strictly to the right
                x1 = min(x1, (bb.x1 + rb.x0) / 2)
            elif rb.x1 <= bb.x0 and y_ov:        # strictly to the left
                x0 = max(x0, (bb.x0 + rb.x1) / 2)
            elif rb.y0 >= bb.y1 and x_ov:        # strictly above
                y1 = min(y1, (bb.y1 + rb.y0) / 2)
            elif rb.y1 <= bb.y0 and x_ov:        # strictly below
                y0 = max(y0, (bb.y0 + rb.y1) / 2)
        cx0, cy0, cx1, cy1 = crop_rect(Bbox([[x0, y0], [x1, y1]]))
        if cx1 - cx0 < 50 or cy1 - cy0 < 50:
            print("SKIP %s_panel%s: rect too small (%d,%d,%d,%d)" % (prefix, letter, cx0, cy0, cx1, cy1))
            continue
        crop = img.crop((cx0, cy0, cx1, cy1))
        p = os.path.join(outdir, "%s_panel%s.png" % (prefix, letter))
        crop.save(p)
        print("PANEL:", p, "rect=(%d,%d,%d,%d)" % (cx0, cy0, cx1, cy1))
        paths.append(p)
    return paths


def main():
    figdir, script, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    outdir = os.path.join(figdir, "panels")
    captured = []
    orig = Figure.savefig

    def patched(self, *a, **k):
        if not captured:
            captured.append(self)
        return orig(self, *a, **k)

    Figure.savefig = patched
    runpy.run_path(script, run_name="__main__")
    # export AFTER the script finished: get_tightbbox returns degenerate boxes
    # while called from inside savefig, but is correct once the run completes
    if captured:
        export_panels(captured[0], outdir, prefix)


if __name__ == "__main__":
    main()
