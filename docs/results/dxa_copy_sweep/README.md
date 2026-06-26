# DXA Copy Sweep Results
# DXA Copy Sweep 结果说明

This directory contains reproducible CSVs and poster plots generated from the existing `dxa_copy` and `dxa_copy_mcast` regression apps.
本目录保存由现有 `dxa_copy` 和 `dxa_copy_mcast` regression app 生成的可复现 CSV 和 poster 图。

All benchmark commands keep L2 enabled with `--l2cache`.
所有 benchmark command 都通过 `--l2cache` 保持 L2 启用。

Do not mix L2-off rows into Figure 3(b), Figure 3(c), or poster claims.
不要把 L2-off 行混入 Figure 3(b)、Figure 3(c) 或 poster 结论。

## Current Official Tables
## 当前正式数据表

Figure 3(b) SimX full matrix:
Figure 3(b) 的 SimX full matrix 数据：

```text
docs/results/dxa_copy_sweep/simx_3b_fullmatrix.csv
```

Coverage is complete at 288 variant rows: 9 hardware points x 16 tile shapes x 2 variants.
覆盖量已完整达到 288 行 variant 数据：9 个 hardware point x 16 个 tile shape x 2 个 variant。

Current status is 288 PASS rows and 0 TIMEOUT rows.
当前状态是 288 行 PASS 和 0 行 TIMEOUT。

The formerly timing-out row `variant=dxa`, `warps=32`, `threads=32`, `tile_rows=16`, `tile_cols=64` now passes with 633245 SimX cycles under `-DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_DXA_ENABLE`.
之前 timeout 的 `variant=dxa`、`warps=32`、`threads=32`、`tile_rows=16`、`tile_cols=64` 行现在在 `-DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_DXA_ENABLE` 下通过，SimX cycles 为 633245。

The root cause was an L2 cache MSHR replay-capacity deadlock exposed by the high-concurrency DXA stream; replay entries now reserve capacity until they either complete or re-enter the MSHR.
root cause 是高并发 DXA stream 暴露的 L2 cache MSHR replay-capacity deadlock；现在 replay entry 会保留容量，直到完成或重新进入 MSHR。

A separate diagnostic CSV with 300-second per-case timeout is:
单独使用每 case 300 秒 timeout 的诊断 CSV 是：

```text
docs/results/dxa_copy_sweep/diagnostics/simx_3b_w32_t32_16_64_timeout300.csv
```

That diagnostic is retained as pre-fix evidence; the official full-matrix CSV has been rerun after the MSHR replay fix.
该诊断作为修复前证据保留；正式 full-matrix CSV 已在 MSHR replay fix 之后重新补跑。

Figure 3(c) SimX full matrix:
Figure 3(c) 的 SimX full matrix 数据：

```text
docs/results/dxa_copy_sweep/simx_3c_fullmatrix.csv
```

Coverage is complete at 288 variant rows, all PASS.
覆盖量已完整达到 288 行 variant 数据，并且全部 PASS。

Each hardware point has 16 per-CTA rows and 16 multicast rows.
每个 hardware point 都有 16 行 per-CTA 和 16 行 multicast。

For Figure 3(c), multicast reduces DXA transfers and GMEM reads by exactly 4x and keeps LMEM writes unchanged.
对于 Figure 3(c)，multicast 会把 DXA transfers 和 GMEM reads 精确降低 4x，并保持 LMEM writes 不变。

The cycle speedup `cycles_percta / cycles_mcast` ranges from about 0.838x to 1.014x, with a mean of about 0.921x, so the current SimX data shows bandwidth-traffic reduction but not cycle-speedup improvement under this setup.
cycle speedup `cycles_percta / cycles_mcast` 约为 0.838x 到 1.014x，均值约 0.921x，因此当前 SimX 数据显示了 bandwidth-traffic reduction，但在这个设置下没有表现出 cycle-speedup improvement。

RTLsim L2-on smoke:
RTLsim 的 L2-on smoke 数据：

```text
docs/results/dxa_copy_sweep/rtlsim_3c_smoke_l2.csv
```

This CSV has 2 PASS rows for the 512x512, 16x16-tile, 4-CTA smoke pair.
该 CSV 有 2 行 PASS，对应 512x512、16x16 tile、4-CTA 的 smoke pair。

The same smoke pair was re-run after regenerating `build/` with `--tooldir=/home/chengxuan99/tools`; the fresh verification CSV is:
同一组 smoke pair 已在用 `--tooldir=/home/chengxuan99/tools` 重新生成 `build/` 后复跑；fresh verification CSV 是：

```text
docs/results/dxa_copy_sweep/rtlsim_3c_smoke_l2_fresh.csv
```

It has 2 PASS rows with the same cycle counts as the official smoke table.
它有 2 行 PASS，cycle 数与正式 smoke 表一致。

RTLsim is not globally broken; the full RTLsim sweep has now been run once, with many Figure 3(c) rows exceeding the 120-second per-case cap.
RTLsim 不是整体跑不通；完整 RTLsim sweep 现在已经跑过一次，但 Figure 3(c) 中很多行超过了每 case 120 秒上限。

Full RTLsim sweeps have now been launched once with the same 120-second per-case cap:
完整 RTLsim sweep 现在已经用相同的每 case 120 秒上限跑过一次：

```text
docs/results/dxa_copy_sweep/rtlsim_3b_fullmatrix_l2.csv
docs/results/dxa_copy_sweep/rtlsim_3c_fullmatrix_l2.csv
```

The Figure 3(b) RTLsim full matrix has 288 rows: 274 PASS and 14 TIMEOUT.
Figure 3(b) 的 RTLsim full matrix 有 288 行：274 行 PASS 和 14 行 TIMEOUT。

The Figure 3(c) RTLsim full matrix has 288 rows: 130 PASS and 158 TIMEOUT.
Figure 3(c) 的 RTLsim full matrix 有 288 行：130 行 PASS 和 158 行 TIMEOUT。

The timeout-heavy Figure 3(c) result should be interpreted as RTLsim runtime coverage under the 120-second cap, not as functional failure evidence by itself; no FAIL rows were produced.
timeout 较多的 Figure 3(c) 结果应解释为 120 秒上限下的 RTLsim runtime coverage，而不是功能失败证据本身；本次没有产生 FAIL 行。

RTLsim full-matrix plots are:
RTLsim full-matrix 图片是：

```text
docs/results/dxa_copy_sweep/rtlsim_fullmatrix_3b_plots/figure3b_speedup.png
docs/results/dxa_copy_sweep/rtlsim_fullmatrix_3c_plots/figure3c_speedup.png
```

These plots mark missing/timeout cells rather than interpolating them.
这些图会标记 missing/timeout cell，而不会插值补齐。

Additional RTLsim partial sweeps:
额外的 RTLsim partial sweeps：

```text
docs/results/dxa_copy_sweep/rtlsim_3b_w8_t8_tiles16_32_l2.csv
docs/results/dxa_copy_sweep/rtlsim_3c_w8_t8_tiles16_32_l2.csv
```

Both partial sweeps are 8 PASS rows for `warps=8`, `threads=8`, and `tile_size in {16,32}^2`.
两组 partial sweep 都是 8 行 PASS，对应 `warps=8`、`threads=8` 和 `tile_size in {16,32}^2`。

The Figure 3(b) RTLsim partial sweep shows DXA speedup from 2.54x to 5.02x over LSU.
Figure 3(b) 的 RTLsim partial sweep 显示 DXA 相对 LSU 的 speedup 为 2.54x 到 5.02x。

The Figure 3(c) RTLsim partial sweep shows multicast cycle speedup from 0.945x to 0.977x while still reducing DXA transfers by 4x; this supports the interpretation that current multicast saves GMEM/DXA traffic but is limited by receiver-side LMEM replay and synchronization/launch overheads.
Figure 3(c) 的 RTLsim partial sweep 显示 multicast cycle speedup 为 0.945x 到 0.977x，同时仍然把 DXA transfers 降低 4x；这支持当前 multicast 节省 GMEM/DXA traffic，但受 receiver-side LMEM replay 以及同步/launch overhead 限制的解释。

CTA-count probe:
CTA 数量 probe：

```text
docs/results/dxa_copy_sweep/simx_3c_numctas8_probe.csv
```

For `warps=8`, `threads=8`, and `tile=16x16`, the 8-CTA multicast row passes, while the 8-CTA per-CTA row times out under the 120-second cap. This suggests that larger CTA groups can make the per-CTA baseline much heavier, but the current multicast implementation still scales receiver LMEM writes with CTA count.
对于 `warps=8`、`threads=8` 和 `tile=16x16`，8-CTA multicast 行通过，而 8-CTA per-CTA 行在 120 秒上限下 timeout。这说明更大的 CTA group 会让 per-CTA baseline 变得更重，但当前 multicast 实现中的 receiver LMEM writes 仍会随 CTA 数量增长。

## Current Plots
## 当前图片

Figure 3(b) plot:
Figure 3(b) 图：

```text
docs/results/dxa_copy_sweep/fullmatrix_3b_plots/figure3b_speedup.png
```

The plot is generated from the complete 288-PASS CSV.
该图由完整的 288-PASS CSV 生成。

Figure 3(c) plot:
Figure 3(c) 图：

```text
docs/results/dxa_copy_sweep/fullmatrix_3c_plots/figure3c_speedup.png
```

## Runner Notes
## Runner 说明

Run from the repository root after configure/build setup:
在 configure/build setup 完成后，从仓库根目录运行：

```bash
python3 tools/dxa/run_copy_sweep.py --driver=simx --figure=both --output docs/results/dxa_copy_sweep/simx_raw.csv
python3 tools/dxa/plot_copy_sweep.py docs/results/dxa_copy_sweep/simx_raw.csv --figure=both
```

For long runs, resume an interrupted CSV with:
长时间运行中断后，可以这样续跑同一个 CSV：

```bash
python3 tools/dxa/run_copy_sweep.py --driver=simx --figure=both --resume --output docs/results/dxa_copy_sweep/simx_raw.csv
```

The runner deliberately reuses existing tests.
runner 会刻意复用现有测试。

Figure 3(b) uses `tests/regression/dxa_copy` with matched LSU and DXA builds.
Figure 3(b) 使用 `tests/regression/dxa_copy`，分别跑 matched LSU 和 DXA builds。

Figure 3(c) uses `tests/regression/dxa_copy_mcast --mode=percta|mcast --num-ctas=4`.
Figure 3(c) 使用 `tests/regression/dxa_copy_mcast --mode=percta|mcast --num-ctas=4`。

By default the runner passes `-DVX_CFG_LMEM_LOG_SIZE=18`.
默认情况下 runner 会传入 `-DVX_CFG_LMEM_LOG_SIZE=18`。

This is required for the requested 4-CTA multicast sweep at 128x128 float tiles: each CTA needs 64 KiB local memory, and four co-resident CTAs need 256 KiB per core.
这是为了支持需求中的 4-CTA multicast 在 128x128 float tile 下运行：每个 CTA 需要 64 KiB local memory，四个 co-resident CTA 需要每个 core 256 KiB。

Use `--lmem-log-size=0` only for explicit diagnostics; do not mix those rows into the official full-matrix CSVs.
只有在明确诊断时才使用 `--lmem-log-size=0`；不要把这些行混入正式 full-matrix CSV。

CSV columns include `figure`, `variant`, `warps`, `threads`, `tile_rows`, `tile_cols`, `status`, `instrs`, `cycles`, `ipc`, `dxa_transfers`, `dxa_gmem_reads`, `dxa_lmem_writes`, `configs`, `command`, and `output_tail`.
CSV 列包括 `figure`、`variant`、`warps`、`threads`、`tile_rows`、`tile_cols`、`status`、`instrs`、`cycles`、`ipc`、`dxa_transfers`、`dxa_gmem_reads`、`dxa_lmem_writes`、`configs`、`command` 和 `output_tail`。

Plot speedups are computed as:
图中的 speedup 计算方式为：

- Figure 3(b): `cycles_lsu / cycles_dxa`
- Figure 3(b)：`cycles_lsu / cycles_dxa`
- Figure 3(c): `cycles_percta / cycles_mcast`
- Figure 3(c)：`cycles_percta / cycles_mcast`
