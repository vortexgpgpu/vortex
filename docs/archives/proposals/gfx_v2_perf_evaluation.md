# gfx_v2 — On-Device Front-End Perf Evaluation (Phase I)

**Scope:** a first SimX measurement of the gfx_v2 on-device draw path — the
device now runs vertex expand + triangle setup + parallel bin-sort + RASTER →
FS → OM with the host only submitting one CP command batch and reading the
final framebuffer ([gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §10
phase «Evaluate»). Characterises where the device cycles go and what bounds
throughput, and frames the on-device front end against the retired host
`Binning()` path.
**Status:** Evaluation — measured, not a design change.
**Date:** 2026-06-12.

---

## 1. Method

The whole draw runs on SimX through the vortexpipe driver (the same path the
graphics suite validates). Aggregate device counters are dumped at teardown via
`vx_device_dump_perf` (`vp_screen_destroy`), enabled by `VORTEX_PROFILING=1`
with a `PERF=1` build:

```
cd build/tests/vulkan/<test> && VORTEX_PROFILING=1 make run-simx PERF=1
```

Config: the default 1-cluster / 1-core / 4-warp × 4-thread SimX build with
RASTER + OM + TEX enabled. Counters are **aggregate over the whole draw** — VS,
expand_k, the nine setup/binning stages, and the fragment shader together; the
dump does not break down per launch (see §4). Each test is one draw into a
small render target, so these numbers characterise the *fixed + per-primitive*
cost of the resident pipeline, not a high-poly steady state.

## 2. Measured results

| Test (target, prims) | instrs | cycles | IPC | inst-mix (alu/lsu/sfu/fpu) | top stall | sched idle | occ |
|---|---:|---:|---:|---|---|---:|---:|
| triangle (64², 1 tri)   |  17 316 |  64 103 | 0.27 | 15 / 21 / 7 / **57** | scrb **89%** | 73% | 98% |
| draw3d   (128², scene)  |  90 492 | 420 955 | 0.22 | 10 / 24 / 4 / **62** | scrb **82%** | 78% | 99% |
| textured (64², quad+TEX)| 126 244 | 417 878 | 0.30 | 19 / 13 / 7 / **61** | scrb **87%** | 70% | 100% |

draw3d memory traffic: loads=57 504, stores=20 624, load_lat≈24 cyc,
divergent-branch rate 9%.

## 3. Analysis

- **FPU / setup-math bound.** 57–62% of issued instructions are FP across all
  three workloads. This is the clip + Q15.16 triangle setup (plane equations
  `(z/w,u/w,v/w,1/w)`, edge equations) in setup_k plus the FS's float work —
  exactly the math the §6.1 front end moved on-device. The binning stages
  themselves (scan / histogram / scatter) are integer and cheap by comparison;
  the FP cost is setup + shading, not the sort.

- **Latency-bound, not throughput-bound.** Scoreboard stalls dominate (82–89%):
  warps wait on long-latency producers — FMA (latency 8) feeding dependent
  setup math, and loads (≈24-cycle latency) for the gathered vertex / bin data.
  Occupancy is ~99% (the CTAs fill the cores) yet IPC is 0.22–0.30, so the
  device is full of *stalled* warps, not idle ones. Fan-out (more warps in
  flight) is the lever, not more cores.

- **Scale.** triangle (1 tri) is the fixed-cost floor (17 K instrs). draw3d's
  scene is ~5× that; textured adds TEX sampling + a heavier FS (most instrs,
  but best IPC — TEX hides some setup latency). The per-primitive cost is
  modest; the front end's setup math, not the bin-sort bookkeeping, is the
  dominant device term.

## 4. On-device front end vs the retired host `Binning()`

Pre-gfx_v2, binning ran on the host CPU and the device only saw RASTER → FS →
OM; the host paid serial `Binning()` time and round-tripped the tile/prim
buffers per draw. gfx_v2 folds setup + binning into the device numbers above
and removes the host work and the per-draw buffer round-trip (now one resident
working set, §6.6, and one CP batch, §6.4).

The numbers above are the **combined** device cost; isolating the front end's
share (and a true device-cycle delta vs the host-binning path) needs a
per-launch breakdown the aggregate dump does not provide. A follow-up harness
would: reset/read perf counters around each launch (expand, setup×3, binning×6,
FS) to attribute cycles per stage; and run the **same** scene through both the
host-`Binning()`→RASTER path and the device front end (the gfx_pipeline_tex
test already renders both — Path A vs Path C — so adding a per-path
`vx_device_dump_perf` there is the natural vehicle). That quantifies «what does
moving binning on-device cost in device cycles» against «what host time + PCIe
round-trip it removes».

## 5. Conclusion

The resident on-device draw path works and its device cost is dominated by FP
triangle-setup math and producer latency, not by the bin-sort. On small scenes
it is latency-bound at ~0.2–0.3 IPC with the cores full — pointing future
tuning at warp fan-out and setup-math latency (FMA scheduling, fixed-point
setup per §3.8) rather than raw core count. The bin-sort bookkeeping the §6.2
redesign added is not a measurable bottleneck. Per-stage attribution and the
host-vs-device delta are the next measurement step (§4).
