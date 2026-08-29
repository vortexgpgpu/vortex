#!/usr/bin/env python3
"""gds_render.py - complete + render an FC placement GDS.

Fusion Compiler streams the design GDS with std-cell PLACEMENTS only (SREFs by
name); the ASAP7 std-cell polygons are not embedded (LEF-only physical lib). This
script merges the ASAP7 cell GDS (drawn at the same 4x scale as the 4x LEF used
for placement, so coordinates align) to produce a complete, self-contained GDS,
then rasterizes a full-chip overview and a zoomed crop with pycairo.

Usage: gds_render.py <design.gds> <out_dir> [zoom_um]
Outputs in <out_dir>: <top>.complete.gds, gds_full.png, gds_zoom.png
"""
import sys, os, math
import gdstk, cairo

DESIGN = sys.argv[1]
OUT    = sys.argv[2]
ZOOM_UM = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0   # window size (4x um)

GDSDIR = os.environ.get("ASAP7_GDS_DIR", "/mnt/nas0/eda.libs/asap7/asap7sc7p5t_28/GDS")
CELLGDS = ["asap7sc7p5t_28_R_220121a.gds", "asap7sc7p5t_28_L_220121a.gds",
           "asap7sc7p5t_28_SL_220121a.gds", "asap7sc7p5t_28_SRAM_220121a.gds"]
# ASAP7 ships GDS at 1x but the 4x-scaled LEF was used for placement, so cells
# must be magnified to match the 4x placement coordinates.
MAG = float(os.environ.get("GDS_CELL_MAG", "4"))

os.makedirs(OUT, exist_ok=True)

# ---- load std-cell defs ----
cellmap = {}
for f in CELLGDS:
    p = os.path.join(GDSDIR, f)
    if not os.path.exists(p):
        continue
    for c in gdstk.read_gds(p).cells:
        cellmap.setdefault(c.name, c)
print("[gds_render] ASAP7 cell defs: %d" % len(cellmap))

# ---- load design ----
d = gdstk.read_gds(DESIGN)
top = d.top_level()[0]
refs = top.references
print("[gds_render] top %s: %d placements" % (top.name, len(refs)))

# ---- merge: add referenced cell defs -> complete GDS ----
used = {}
for r in refs:
    n = r.cell if isinstance(r.cell, str) else r.cell.name
    used[n] = used.get(n, 0) + 1
resolved = [n for n in used if n in cellmap]
missing  = [n for n in used if n not in cellmap]
print("[gds_render] distinct cells %d  resolved %d  missing %d" % (len(used), len(resolved), len(missing)))
if missing:
    print("[gds_render] missing:", sorted(missing)[:20])
# scale each cell's geometry by MAG and add under the same name, so the design's
# SREFs (magnification 1) resolve to correctly-sized cells -> consistent GDS.
for n in resolved:
    src = cellmap[n]
    nc = gdstk.Cell(n)
    nc.add(*[p.scale(MAG, center=(0, 0)) for p in src.get_polygons(depth=None)])
    d.add(nc)
complete = os.path.join(OUT, top.name + ".complete.gds")
d.write_gds(complete)
print("[gds_render] wrote %s (%d bytes)" % (complete, os.path.getsize(complete)))

# ---- layer color map (by GDS layer number) ----
LAYER_RGB = {}
_palette = [(0.20,0.45,0.95),(0.95,0.30,0.25),(0.20,0.75,0.35),(0.95,0.80,0.20),
            (0.30,0.80,0.85),(0.85,0.35,0.85),(0.95,0.55,0.20),(0.70,0.70,0.75),
            (0.55,0.85,0.45),(0.45,0.55,0.90),(0.90,0.50,0.55),(0.60,0.40,0.85)]
def layer_color(layer):
    if layer not in LAYER_RGB:
        LAYER_RGB[layer] = _palette[len(LAYER_RGB) % len(_palette)]
    return LAYER_RGB[layer]

def vt_color(name):
    if name.endswith("_SRAM"): return (0.95,0.55,0.20)
    if name.endswith("_SL"):   return (0.95,0.30,0.25)
    if name.endswith("_L"):    return (0.20,0.75,0.35)
    return (0.20,0.45,0.95)  # _R

(x0,y0),(x1,y1) = top.bounding_box()
W = x1-x0; H = y1-y0

# ---- full-chip overview: cell footprints colored by Vt flavor ----
def render_full(png, px=2000):
    sc = px/max(W,H)
    wpx, hpx = int(W*sc)+2, int(H*sc)+2
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, wpx, hpx)
    cr = cairo.Context(surf)
    cr.set_source_rgb(0.06,0.06,0.08); cr.paint()
    # cache cell footprint sizes
    size = {}
    for n in resolved:
        (a,b),(c,e) = cellmap[n].bounding_box()
        size[n] = ((c-a)*MAG, (e-b)*MAG)
    for r in refs:
        n = r.cell if isinstance(r.cell,str) else r.cell.name
        if n not in size: continue
        w,h = size[n]
        rot = (r.rotation or 0.0)
        if abs(math.sin(rot)) > 0.5: w,h = h,w   # 90/270 swap
        ox,oy = r.origin
        X = (ox-x0)*sc; Y = hpx-(oy-y0)*sc - h*sc   # flip Y
        cr.set_source_rgb(*vt_color(n))
        cr.rectangle(X, Y, max(w*sc,0.4), max(h*sc,0.4)); cr.fill()
    surf.write_to_png(png)
    print("[gds_render] wrote %s (%dx%d)" % (png, wpx, hpx))

# ---- zoom: real polygons in a central window, colored by layer ----
def render_zoom(png, win_um, px=1600):
    cx,cy = (x0+x1)/2.0, (y0+y1)/2.0
    wx0,wy0,wx1,wy1 = cx-win_um/2, cy-win_um/2, cx+win_um/2, cy+win_um/2
    sub = gdstk.Cell("__zoom__")
    cnt = 0
    for r in refs:
        ox,oy = r.origin
        if wx0-5 <= ox <= wx1+5 and wy0-5 <= oy <= wy1+5:
            n = r.cell if isinstance(r.cell,str) else r.cell.name
            if n in cellmap:
                sub.add(gdstk.Reference(cellmap[n], origin=r.origin,
                        rotation=r.rotation or 0, x_reflection=bool(r.x_reflection),
                        magnification=MAG))
                cnt += 1
    sub.flatten()
    polys = sub.get_polygons(depth=None)
    # top-level routing (paths/wires live in the top cell, already in 4x space)
    route = []
    try:
        for p in top.get_polygons(depth=0, include_paths=True):
            (px0,py0),(px1,py1) = p.bounding_box()
            if px1 >= wx0 and px0 <= wx1 and py1 >= wy0 and py0 <= wy1:
                route.append(p)
    except Exception as e:
        print("[gds_render] top-level routing: %s" % e)
    polys = polys + route
    print("[gds_render] zoom window %.1fum: %d cells, %d cell-polys + %d routing-polys" % (win_um, cnt, len(polys)-len(route), len(route)))
    sc = px/win_um
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, px, px)
    cr = cairo.Context(surf)
    cr.set_source_rgb(0.04,0.04,0.05); cr.paint()
    # draw lower layers first (sort by layer)
    polys.sort(key=lambda p: p.layer)
    for p in polys:
        pts = p.points
        if len(pts) < 3: continue
        cr.set_source_rgba(*layer_color(p.layer), 0.78)
        cr.move_to((pts[0][0]-wx0)*sc, px-(pts[0][1]-wy0)*sc)
        for (ux,uy) in pts[1:]:
            cr.line_to((ux-wx0)*sc, px-(uy-wy0)*sc)
        cr.close_path(); cr.fill()
    surf.write_to_png(png)
    print("[gds_render] wrote %s (%dx%d)" % (png, px, px))

render_full(os.path.join(OUT, "gds_full.png"))
render_zoom(os.path.join(OUT, "gds_zoom.png"), ZOOM_UM)
print("[gds_render] DONE")
