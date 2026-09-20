#!/usr/bin/env python3
"""Parse improved-architecture sweep logs and regenerate the report plots.

Reads the original results.json (pre-fix rtlsim sweep) and the improved/
logs, writes results_improved.json, and regenerates the five report PNGs
in the original visual style with the improved DXA/FEDP2K series.
"""
import json, re, os, sys, glob

BASE = os.path.expanduser('~/dev/vortex_spg/eval_tcu_vs_wg')
IMP = os.path.join(BASE, 'improved')

def parse_log(path):
    d = {}
    txt = open(path, errors='replace').read()
    m = None
    for m in re.finditer(r'instrs=(\d+), cycles=(\d+), IPC=([\d.]+)', txt): pass
    if not m: return None
    d['instrs'], d['cycles'] = int(m.group(1)), int(m.group(2))
    d['ipc'] = float(m.group(3))
    m = re.search(r'scheduler: idle=(\d+)%', txt)
    if m: d['idle'] = int(m.group(1))
    m = re.search(r'stalls: fetch=\d+%, ibuf=\d+%, scrb=(\d+)%', txt)
    if m: d['stall_scrb'] = int(m.group(1))
    m = re.search(r'memory: ifetch_lat=[\d.]+, load_lat=([\d.]+), loads=(\d+), stores=(\d+)', txt)
    if m:
        d['load_lat'], d['loads'], d['stores'] = float(m.group(1)), int(m.group(2)), int(m.group(3))
    d['passed'] = 'PASSED!' in txt
    d['flop_per_cycle'] = round(4194304.0 / d['cycles'], 3)
    return d

def load_improved():
    rows = []
    names = {'dxa2k':'sgemm_tcu_wg_dxa_2k','dxa':'sgemm_tcu_wg_dxa',
             'wmma':'sgemm_tcu','wg':'sgemm_tcu_wg'}
    for f in glob.glob(os.path.join(IMP, '*_nt*_iw*.log')):
        b = os.path.basename(f)
        m = re.match(r'(dxa2k|dxa|wmma|wg)_nt(\d+)_iw(\d+)\.log', b)
        if not m: continue
        d = parse_log(f)
        if d is None: continue
        d['app'] = names[m.group(1)]
        d['nt'], d['iw'] = int(m.group(2)), int(m.group(3))
        rows.append(d)
    return rows

def main():
    orig = json.load(open(os.path.join(BASE, 'results.json')))
    imp = load_improved()
    json.dump(imp, open(os.path.join(IMP, 'results_improved.json'), 'w'), indent=1)

    def get(rows, app, nt, iw, field='cycles'):
        for r in rows:
            if r['app'] == app and r['nt'] == nt and r['iw'] == iw:
                return r.get(field)
        # NT=32 baseline kernels were measured with the improved sweep only;
        # never backfill the DXA variants (no pre-fix data exists at NT=32)
        if rows is orig and app in ('sgemm_tcu', 'sgemm_tcu_wg'):
            for r in imp:
                if r['app'] == app and r['nt'] == nt and r['iw'] == iw:
                    return r.get(field)
        return None

    cfgs = [(4,1),(4,2),(4,4),(8,1),(8,2),(8,4),(16,1),(16,2),(16,4),(32,1),(32,2),(32,4)]
    print(f"{'cfg':>10s} {'wmma':>10s} {'dxa-old':>10s} {'dxa-new':>10s} {'2k-old':>10s} {'2k-new':>10s}")
    for nt, iw in cfgs:
        row = [get(orig,'sgemm_tcu',nt,iw), get(orig,'sgemm_tcu_wg_dxa',nt,iw),
               get(imp,'sgemm_tcu_wg_dxa',nt,iw), get(orig,'sgemm_tcu_wg_dxa_2k',nt,iw),
               get(imp,'sgemm_tcu_wg_dxa_2k',nt,iw)]
        print(f"NT{nt:>3}/IW{iw} " + " ".join(f"{v:>10,}" if v else f"{'—':>10s}" for v in row))

    if '--plots' not in sys.argv:
        return
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C = {'wmma':'#2563eb','wg':'#f59e0b','dxa':'#0d9488','k2':'#9333ea','old':'#b0b7c3'}
    labels = [f"NT={nt}\nIW={iw}" for nt,iw in cfgs]
    X = np.arange(len(cfgs))

    def series(rows, app, field='cycles'):
        return [get(rows, app, nt, iw, field) for nt,iw in cfgs]

    wmma  = series(orig,'sgemm_tcu')
    wg    = series(orig,'sgemm_tcu_wg')
    dxa_o = series(orig,'sgemm_tcu_wg_dxa')
    dxa_n = series(imp,'sgemm_tcu_wg_dxa')
    k2_n  = series(imp,'sgemm_tcu_wg_dxa_2k')

    def fmt(v):
        if v is None: return ''
        return f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"

    def bars(ax, offs, vals, w, color, label, log=False, annotate=True, alpha=1.0, hatch=None):
        xs, ys = [], []
        for i, v in enumerate(vals):
            if v is None: continue
            xs.append(X[i]+offs); ys.append(v)
        b = ax.bar(xs, ys, w, color=color, label=label, alpha=alpha, hatch=hatch,
                   edgecolor='white' if hatch is None else color, linewidth=0.3)
        if annotate:
            for x, y in zip(xs, ys):
                ax.text(x, y*1.04 if log else y+ax.get_ylim()[1]*0.005, fmt(y),
                        ha='center', va='bottom', fontsize=7.5)
        return b

    # ---- cycles ----
    fig, ax = plt.subplots(figsize=(16.2, 5.8), dpi=100)
    w = 0.17
    bars(ax,-2*w, wmma, w, C['wmma'], 'WMMA', log=True)
    bars(ax,-1*w, wg,   w, C['wg'],   'WGMMA', log=True)
    bars(ax, 0*w, dxa_o,w, C['old'],  'WGMMA+DXA (pre-fix)', log=True, annotate=False)
    bars(ax, 1*w, dxa_n,w, C['dxa'],  'WGMMA+DXA (improved)', log=True)
    bars(ax, 2*w, k2_n, w, C['k2'],   '+FEDP2K (improved)', log=True)
    ax.set_yscale('log'); ax.set_ylabel('total cycles (log)')
    ax.set_title('Total cycles — 128×128×128 fp16→fp32, rtlsim, NW=16 — improved architecture (FEDP2K: NT=4 unsupported)')
    ax.set_xticks(X); ax.set_xticklabels(labels)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3, which='both')
    for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(IMP,'plot_cycles.png')); plt.close(fig)

    # ---- flops ----
    fig, ax = plt.subplots(figsize=(16.2, 5.4), dpi=100)
    def fl(v): return [4194304.0/x if x else None for x in v]
    for offs,vals,c,l in [(-2*w,fl(wmma),C['wmma'],'WMMA'),(-1*w,fl(wg),C['wg'],'WGMMA'),
                          (0,fl(dxa_o),C['old'],'WGMMA+DXA (pre-fix)'),
                          (1*w,fl(dxa_n),C['dxa'],'WGMMA+DXA (improved)'),
                          (2*w,fl(k2_n),C['k2'],'+FEDP2K (improved)')]:
        xs=[X[i]+offs for i,v in enumerate(vals) if v]; ys=[v for v in vals if v]
        ax.bar(xs, ys, w, color=c, label=l)
        if c != C['old']:
            for x,y in zip(xs,ys): ax.text(x, y+0.5, f"{y:.1f}", ha='center', fontsize=7.5)
    ax.set_ylabel('FLOP / cycle'); ax.set_title('Throughput — FLOP per cycle (higher is better)')
    ax.set_xticks(X); ax.set_xticklabels(labels); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(IMP,'plot_flops.png')); plt.close(fig)

    # ---- speedup vs wmma (now ratio; <1 = faster than WMMA) ----
    fig, ax = plt.subplots(figsize=(16.2, 5.4), dpi=100)
    def ratio(vals): return [ (v/wm if (v and wm) else None) for v, wm in zip(vals, wmma)]
    for offs,vals,c,l in [(-1.5*w,ratio(wg),C['wg'],'WGMMA / WMMA'),
                          (-0.5*w,ratio(dxa_o),C['old'],'WGMMA+DXA pre-fix / WMMA'),
                          (0.5*w,ratio(dxa_n),C['dxa'],'WGMMA+DXA improved / WMMA'),
                          (1.5*w,ratio(k2_n),C['k2'],'+FEDP2K improved / WMMA')]:
        xs=[X[i]+offs for i,v in enumerate(vals) if v]; ys=[v for v in vals if v]
        ax.bar(xs, ys, w, color=c, label=l)
        if c != C['old']:
            for x,y in zip(xs,ys): ax.text(x, y+0.15, f"{y:.2f}×", ha='center', fontsize=7.5)
    ax.axhline(1.0, color='#555', ls='--', lw=1); ax.text(X[-1]+0.45, 1.0, 'parity', fontsize=8, va='center')
    ax.set_ylabel('cycles relative to WMMA'); ax.set_title('Cycles vs WMMA (1.0 = parity; below = faster than WMMA)')
    ax.set_xticks(X); ax.set_xticklabels(labels); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(IMP,'plot_speedup.png')); plt.close(fig)

    # ---- instr + scrb stalls ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18.0, 5.4), dpi=100)
    ins = lambda rows, app: series(rows, app, 'instrs')
    for offs,vals,c,l in [(-1.5*w,ins(orig,'sgemm_tcu'),C['wmma'],'WMMA'),
                          (-0.5*w,ins(orig,'sgemm_tcu_wg'),C['wg'],'WGMMA'),
                          (0.5*w,ins(imp,'sgemm_tcu_wg_dxa'),C['dxa'],'WGMMA+DXA (improved)'),
                          (1.5*w,ins(imp,'sgemm_tcu_wg_dxa_2k'),C['k2'],'+FEDP2K (improved)')]:
        xs=[X[i]+offs for i,v in enumerate(vals) if v]; ys=[v for v in vals if v]
        a1.bar(xs, ys, w, color=c, label=l)
    a1.set_yscale('log'); a1.set_ylabel('retired instructions (log)')
    a1.set_title('Dynamic instruction count'); a1.legend(fontsize=9)
    a1.set_xticks(X); a1.set_xticklabels(labels, fontsize=8); a1.grid(axis='y', alpha=0.3, which='both')
    st = lambda rows, app: series(rows, app, 'stall_scrb')
    for offs,vals,c,l in [(-1.5*w,st(orig,'sgemm_tcu'),C['wmma'],'WMMA'),
                          (-0.5*w,st(orig,'sgemm_tcu_wg'),C['wg'],'WGMMA'),
                          (0.5*w,st(imp,'sgemm_tcu_wg_dxa'),C['dxa'],'WGMMA+DXA (improved)'),
                          (1.5*w,st(imp,'sgemm_tcu_wg_dxa_2k'),C['k2'],'+FEDP2K (improved)')]:
        xs=[X[i]+offs for i,v in enumerate(vals) if v is not None]; ys=[v for v in vals if v is not None]
        a2.bar(xs, ys, w, color=c, label=l)
    a2.set_ylabel('scoreboard stall %'); a2.set_title('Scoreboard (dependency) stalls')
    a2.set_xticks(X); a2.set_xticklabels(labels, fontsize=8); a2.legend(fontsize=9); a2.grid(axis='y', alpha=0.3)
    for ax_ in (a1,a2):
        for s in ('top','right'): ax_.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(IMP,'plot_instrs_stalls.png')); plt.close(fig)

    # ---- scaling (log-log lines) ----
    fig, ax = plt.subplots(figsize=(10.4, 6.0), dpi=100)
    NTS = [4, 8, 16, 32]
    fam = [('sgemm_tcu', orig, C['wmma'], 'WMMA'), ('sgemm_tcu_wg', orig, C['wg'], 'WGMMA'),
           ('sgemm_tcu_wg_dxa', imp, C['dxa'], 'WGMMA+DXA impr.'),
           ('sgemm_tcu_wg_dxa_2k', imp, C['k2'], '+FEDP2K impr.')]
    marks = {1:'o', 2:'s', 4:'^'}
    for app, rows, c, l in fam:
        for iw in (1,2,4):
            ys = [get(rows, app, nt, iw) for nt in NTS]
            pts = [(nt,y) for nt,y in zip(NTS,ys) if y]
            if not pts: continue
            alpha = {1:0.35, 2:0.65, 4:1.0}[iw]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marks[iw],
                    color=c, alpha=alpha, label=f"{l} IW={iw}", lw=2)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks(NTS); ax.set_xticklabels([str(n) for n in NTS])
    ax.set_xlabel('NUM_THREADS'); ax.set_ylabel('total cycles (log)')
    ax.set_title('Warp-width scaling — improved architecture')
    ax.legend(fontsize=7.5, ncol=4); ax.grid(alpha=0.3, which='both')
    for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(IMP,'plot_scaling.png')); plt.close(fig)
    print("plots written")

if __name__ == '__main__':
    main()
