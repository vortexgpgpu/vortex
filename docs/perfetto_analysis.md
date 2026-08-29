# Vortex Perfetto Trace: Performance Analysis Guide

This guide explains how to **generate** and **interpret** a Vortex GPU trace that has been converted into **Perfetto-compatible Chrome Trace JSON** using `vortex_perfetto.py`.

It is written for people who are:
- New to Perfetto
- New to Vortex trace semantics
- Trying to identify performance bottlenecks (pipeline stalls, memory pressure, low warp activity)

---

## What you will see in Perfetto

When you load an exported trace in Perfetto, you will typically see:
- **Global Legacy Events** (metadata/trace-format bookkeeping)
- **Vortex GPU 1** (the main process containing all GPU timelines)

This guide focuses on **Vortex GPU 1**.

---

## Generate and open a Perfetto trace

### Prerequisites

- Python 3.8+
- An input log file:
  - **SimX log** (default log type), or
  - **RTL sim log** (`run_rtlsim.log`, `run.log`, etc.)

### Convert the log

#### SimX (default)

```bash
python3 vortex_perfetto.py run_simx.log -t simx -c -o vortex.perfetto.json.gz
```

#### RTL simulation

```bash
python3 vortex_perfetto.py run_rtlsim.log -t rtlsim -c -o vortex.perfetto.json.gz
```

### Common options

- **Limit export to a cycle/tick window** (highly recommended for huge logs):

```bash
python3 vortex_perfetto.py run_rtlsim.log -t rtlsim \
  --cycle-min 10000 --cycle-max 20000 -c -o window.json.gz
```

- **Provide a time base** (map cycles/ticks to microseconds):
  - `--freq-mhz <MHz>` or
  - `--cycle-ns <ns>` (overrides `--freq-mhz`)

- **Value capture** (`--values`):
  - `none`: no register values
  - `dest`: destination value only (attached to the **commit** stage marker)
  - `all`: source values at dispatch/operands + destination value at commit

- **RTL only: start-of-trace gating**
  - By default, RTL parsing begins only after the exporter sees `[VXDRV] START:`.
  - If your log doesn’t contain that line, use `--no-vxdrv-start`.

- **Uop parent→child flow arrows**
  - If your trace includes `parent=#<uuid>` fields, enable `--parent-flow` to draw arrows between parent and uop events.

### Output files

- `*.json.gz` (recommended): gzip-compressed Chrome Trace JSON
- `*.json`: uncompressed

### Open in Perfetto

- **Web UI**: open the Perfetto UI and load the `.json` or `.json.gz` file.
- **VS Code**: use a Perfetto trace viewer extension to open the exported file.

---

## Vortex GPU 1: track organization

Perfetto groups timelines by **process**. The exporter creates a single process:

- **Vortex GPU 1**

Within that process, the exporter creates multiple **tracks** (Perfetto “threads”) whose names encode hierarchy.

### Naming scheme

Track names are derived from the module path and (when present) the `(cluster, socket, core)` identity:
- Base name: `cluster<i>-socket<j>-core<k>` (or `global` if not present)
- Warp tracks: `cluster0-socket0-core0: warp3`
- Warp-state tracks: `cluster0-socket0-core0: warp3 state`
- Memory/cache tracks: `cluster0-socket0-core0: dcache` / `...: l2` / `...: mem`, etc.

Use Perfetto search to jump quickly:
- `core0` / `core3`
- `warp0` / `warp7`
- `dcache`, `icache`, `l2`, `l3`, `mem`

---

## What the exporter emits

### 1) Warp instruction lifetimes

Each instruction instance is keyed by **UUID** and emitted as an **async slice** that begins at the first observed stage and ends at **commit**.

On the same warp track, the exporter also emits **instant stage markers** (e.g., `schedule`, `decode`, `dispatch`, `execute`, `commit`).

When you click a stage marker, you may see fields like:
- `uuid`, `wid`, `PC`, `tmask`, `ex`, `op`, `rd/rs1/rs2/rs3`, `sop/eop`
- If `--values dest` or `--values all`: `dest_value` appears **only at commit**

### 2) Warp state counters

Scheduler warp-state updates are emitted as counters on a per-warp “state” track:
- `active`
- `stalled`
- `active_threads` (derived from `tmask`)

These are useful for quickly spotting underutilization vs widespread stalling.

### 3) Cache/memory events

The exporter emits **instant markers** for cache/memory activity when it can recognize a cache-like line (hit/miss/fill/writeback/req/rsp/replay/mshr, etc.).

These appear on per-level tracks (icache/dcache/l2/l3/mem/unknown) and are labeled like:
- `dcache:miss`
- `l2:hit`
- `mem:req`

with arguments (when present) such as `uuid`, `wid`, `PC`, `tmask`, `addr`, `tag`.

**Important:** the exporter does **not** attempt to reconstruct full request lifetimes or include large payloads; heavy fields are intentionally stripped to keep JSON size manageable.

### 4) Raw “unmapped” instants

If the parser sees a UUID-tagged line but can’t map it to a known stage, it is still emitted as a lightweight “raw” instant event so it remains searchable and visible.

---

## Recommended analysis workflows

### 1) Find long-latency instructions

1. Expand **Vortex GPU 1**.
2. Search for a warp track, e.g. `cluster0-socket0-core0: warp0`.
3. Zoom into a region with low throughput.
4. Look for a very long async instruction slice (large time span).
5. Click the slice (or the nearby `commit` marker) and note the `uuid`, `op`, `ex`, `tmask`, and `PC`.

### 2) Correlate an instruction with cache/memory activity

If cache/memory events include `uuid`, you can correlate them to a long-latency instruction:

1. From the long instruction, copy its `uuid`.
2. Use Perfetto search to find events containing that `uuid`.
3. Inspect cache/memory instants across levels (`dcache`, `l2`, `l3`, `mem`) and compare how many events show up and where.

This is a “best effort” correlation: some cache/memory lines may not carry UUIDs, especially for background activity (fills/writebacks/evictions).

### 3) Check whether the machine is underutilized vs stalled

Use warp-state counters:
- **active low** across warps: not enough runnable work / warps disabled / short kernel.
- **stalled high** across warps: waiting on dependencies or long-latency events.
- **active_threads low**: divergence or masking is limiting effective throughput.

### 4) Sanity-check thread-mask interpretation (RTL vs SimX)

The exporter normalizes RTL `tmask` bit ordering to match the SimX convention (`bit0 == thread0`).

---

## Glossary

- **UUID**: Unique instruction instance id used to correlate events.
- **WID**: Warp id.
- **tmask**: Thread mask (active lanes).
- **Stages**: Stage markers like `schedule`, `decode`, `dispatch`, `execute`, `commit` emitted as instant events on a warp track.
- **Cache levels**: `icache`, `dcache`, `l2`, `l3`, `mem` (inferred from the module path).
- **Flow arrow**: Optional parent→uop correlation when `--parent-flow` is enabled.

---

## Support / feedback

If you want help interpreting a bottleneck, share:
- The cycle window of interest (`--cycle-min/--cycle-max`)
- A screenshot of the relevant warp + memory tracks
- The UUID(s) of the long-latency instruction(s)
