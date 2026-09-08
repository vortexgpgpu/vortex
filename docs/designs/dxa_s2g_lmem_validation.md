# S2G LMEM credit/reorder validation — 2026-09-08

These are short **Verilator RTL worker unit tests**, using memory endpoint
stubs. They do not run SimX, a GPU kernel, a cache hierarchy, or an FPGA.
Consequently they establish neither L2 behavior nor kernel speedup. SimX
timing parity and end-to-end kernel measurements remain outstanding.

The base before these uncommitted changes was `a7e97ec5a` on
`cfx/s2g-throughput`. No existing source-tree build outputs were reused.
Each configuration below was configured in its own `/tmp` directory with:

```sh
/home/chengxuan99/tmp_test/kitsune/repos/vortex-master/worktrees/cfx-s2g-throughput/configure --xlen=32 --tooldir=/home/chengxuan99/tools
```

Every run then used this command, with the additional flags from the table
appended inside `CONFIGS`:

```sh
make -C hw/unittest/dxa_s2g VERILATOR_PATH=/opt/verilator THREADS=2 CONFIGS='-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_DXA_GROUP_ENABLE -DVX_CFG_EXT_DXA_S2G_ENABLE <additional flags>' > final-build.log 2>&1
hw/unittest/dxa_s2g/dxa_s2g > final-test.log 2>&1
```

All seven final build and executable commands returned zero, with no Verilator
warnings. The table abbreviates `-DVX_CFG_DXA_S2G_PIPELINED` as **P** and
`-DVX_CFG_DXA_S2G_PIPE_MULTI_READ` as **M**; all line sizes are 64 B.

| Additional flags | Checks | Failed | Build directory |
|---|---:|---:|---|
| None (legacy) | 643935 | 0 | `/tmp/vortex-s2g-legacy-20260908.DG61rJ` |
| P (one read credit) | 637493 | 0 | `/tmp/vortex-s2g-single-20260908.AbrX54` |
| P M (4 slots, 4 credits, 16 B LMEM word) | 624110 | 0 | `/tmp/vortex-s2g-ooo-20260908.YstaRI` |
| P M `-DVX_CFG_LMEM_NUM_BANKS=1 -DVX_CFG_DXA_S2G_PIPE_SLOTS=3 -DVX_CFG_DXA_S2G_READ_CREDITS=4` | 630826 | 0 | `/tmp/vortex-s2g-word4-20260908.GrJGT6` |
| P M `-DVX_CFG_LMEM_NUM_BANKS=64` (256 B LMEM word) | 626712 | 0 | `/tmp/vortex-s2g-word256-20260908.3hJSQR` |
| P M `-DVX_CFG_DXA_S2G_PIPE_SLOTS=8` | 620132 | 0 | `/tmp/vortex-s2g-credit8-20260908.EhIMv2` |
| P M `-DVX_CFG_DXA_S2G_PIPE_SLOTS=1` | 634687 | 0 | `/tmp/vortex-s2g-credit1-20260908.Kvv4IG` |

The 4 B LMEM case requires **17** source words for an unaligned 64 B line, so
it exercises a 5-bit word tag rather than the old fixed 4-bit field. It also
checks non-power-of-two slot wrap/reuse. The 256 B LMEM case exercises a source
offset larger than a cache line and the two-word counter boundary.

In the default multi-read run the 24-row case emits all 24 stores, reaches four
pending reads, and observes four reordered responses. Its unaligned 20-line
case observes 18 reordered responses. With eight slots/credits, those cases
reach eight pending reads and observe 11 and 37 reordered responses.

Each run validates all 65536 destination bytes after every transfer (including
untouched canaries). It checks request fields/data remain stable throughout
LMEM/GMEM backpressure, exactly one source event, and no further source reads
after the event. SMEM is overwritten immediately at the accepted source event;
destination verification still uses the original source image. The OOB-tail
case deliberately holds global stores and checks source completion precedes
their acceptance. Counts include repeated per-byte/per-cycle checks, not
hundreds of thousands of distinct randomized scenarios.

## Negative control

The original `VX_gpu_pkg.sv` and `VX_dxa_s2g_data_pipe.sv` from `a7e97ec5a`
were copied into
`/tmp/vortex-s2g-old-repro-20260908.WuZD02/original` without changing the
worktree. The same strengthened harness was built with P M and these overrides:

```sh
PARAMS='-I/tmp/vortex-s2g-old-repro-20260908.WuZD02/original'
RTL_PKGS='/tmp/vortex-s2g-old-repro-20260908.WuZD02/original/VX_gpu_pkg.sv /home/chengxuan99/tmp_test/kitsune/repos/vortex-master/worktrees/cfx-s2g-throughput/hw/rtl/VX_trace_pkg.sv /home/chengxuan99/tmp_test/kitsune/repos/vortex-master/worktrees/cfx-s2g-throughput/hw/rtl/dxa/VX_dxa_pkg.sv'
```

Its `test.log` records `checks=604659 failed=168`, including:

```text
LMEM request changed while stalled
GMEM request changed while stalled
GMEM payload changed while stalled
store beats got=21 want=24
```

The previous short tests did not reveal these bugs. The negative control
shows why delayed responses, transfers larger than the payload pool, and
backpressure checks are required before enabling this path for kernel tuning.

## Scope still unverified

- Matching SimX timing changes, full-core/NoC tag routing, and model parity.
- Kernel performance/utilization on `orcas2.cs.ucla.edu` with L2 enabled.
- Area, timing, and registered-interface retiming.
- Invalid/stale LMEM response recovery: malformed duplicates assert; the
  memory protocol is assumed to deliver one response per accepted request.

No cache visibility, destination/full-wait, or S2G multicast behavior is added.
