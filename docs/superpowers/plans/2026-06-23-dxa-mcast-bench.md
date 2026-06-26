# DXA Copy And 4-CTA Multicast Benchmark Re-Engineering Plan
# DXA Copy 与 4-CTA Multicast Benchmark 重构计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.
> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，按任务逐项执行本计划。

> **Tracking rule:** Steps use checkbox (`- [ ]`) syntax and should be checked only after local verification succeeds.
> **跟踪规则：** 每一步使用 checkbox (`- [ ]`) 语法，只有本地验证通过后才勾选。

**Goal:** Produce reproducible Figure 3(b) and Figure 3(c) data without duplicating existing DXA tests.
**目标：** 在不重复造 DXA 测试的前提下，产出可复现的 Figure 3(b) 和 Figure 3(c) 数据。

**Architecture:** Reuse `tests/regression/dxa_copy` for Figure 3(b), because it already has LSU-versus-DXA compilation paths and N-D tile controls.
**架构：** Figure 3(b) 复用 `tests/regression/dxa_copy`，因为它已经具备 LSU-versus-DXA 编译路径和 N-D tile 控制。

**Architecture:** Re-engineer `tests/regression/dxa_copy_mcast` for Figure 3(c), because it already validates intra-core multicast semantics but needs benchmark modes, fixed 4-CTA grouping, arbitrary 2D tile shapes, and a per-CTA DXA baseline.
**架构：** Figure 3(c) 重构 `tests/regression/dxa_copy_mcast`，因为它已经验证 intra-core multicast 语义，但还需要 benchmark mode、固定 4-CTA grouping、任意 2D tile shape，以及 per-CTA DXA baseline。

**Architecture:** Do not create `tests/regression/dxa_mcast_bench` unless extending `dxa_copy_mcast` would make the existing correctness smoke test unreadable.
**架构：** 不默认新建 `tests/regression/dxa_mcast_bench`，除非扩展 `dxa_copy_mcast` 会让现有 correctness smoke test 变得不可读。

**Tech Stack:** Vortex runtime v2 launch API, Vortex kernel C++ intrinsics, DXA descriptor helpers, `vortex::barrier`, `vortex::group_barrier`, SimX and RTLsim via generated `build/ci/blackbox.sh`, and Python 3 for sweeps and plotting.
**技术栈：** 使用 Vortex runtime v2 launch API、Vortex kernel C++ intrinsics、DXA descriptor helpers、`vortex::barrier`、`vortex::group_barrier`、通过生成的 `build/ci/blackbox.sh` 跑 SimX 和 RTLsim，并用 Python 3 做 sweep 与画图。

---

## Non-Duplication Rule
## 不重复劳动规则

Every new benchmark requirement must first be mapped to an existing regression app.
每一个新的 benchmark 需求都必须先映射到一个现有 regression app。

Only add a new app after documenting the exact existing app that cannot be extended cleanly.
只有在明确记录某个现有 app 无法干净扩展之后，才允许新增 app。

Prefer adding modes, CLI flags, stable summary lines, and external runners over cloning kernels.
优先添加 mode、CLI 参数、稳定 summary line 和外部 runner，而不是复制 kernel。

Keep correctness smoke behavior available after benchmark re-engineering.
benchmark 重构之后，仍然要保留 correctness smoke 行为。

Even when re-engineering is necessary, reuse existing code paths before writing new host or kernel code.
即使确实需要 re-engineering，也要先复用现有 code path，再考虑写新的 host 或 kernel code。

Prefer parameterizing the current code over cloning it because this saves review tokens and reduces the chance of subtle host/kernel divergence.
优先参数化当前代码，而不是 clone 一份，因为这样节省 review token，也降低 host/kernel 出现隐蔽分歧的概率。

Treat re-engineering as a compatibility layer around the existing smoke and DXA paths, not as permission to replace working host/kernel code wholesale.
把 re-engineering 当成包在现有 smoke 和 DXA 路径外面的兼容层，而不是把可工作的 host/kernel 代码整套替换掉的许可。

If code must be shared between smoke and benchmark modes, extract the smallest helper that preserves the original behavior.
如果 smoke mode 和 benchmark mode 必须共享代码，就抽出最小 helper，并保持原始行为不变。

If a new file is proposed, the plan must name the exact existing function or block that cannot be safely reused.
如果计划新增文件，必须明确指出哪个现有 function 或代码块无法安全复用。

---

## Existing Test Inventory
## 现有测试盘点

`tests/regression/dxa_copy` is the Figure 3(b) starting point.
`tests/regression/dxa_copy` 是 Figure 3(b) 的起点。

It already supports LSU mode when DXA is not enabled and DXA mode when `VX_CFG_EXT_DXA_ENABLE` is enabled.
它已经支持未启用 DXA 时的 LSU mode，以及启用 `VX_CFG_EXT_DXA_ENABLE` 时的 DXA mode。

It already supports N-D size and tile flags such as `-d2 -s0 512 -s1 512 -t0 16 -t1 16`.
它已经支持 N-D size 和 tile 参数，例如 `-d2 -s0 512 -s1 512 -t0 16 -t1 16`。

It should not be forked for Figure 3(b).
Figure 3(b) 不应该 fork 它。

`tests/regression/dxa_copy_mcast` is the Figure 3(c) starting point.
`tests/regression/dxa_copy_mcast` 是 Figure 3(c) 的起点。

It already programs a multicast descriptor, sets `cluster_dim[0]`, uses `vortex::dxa_multicast_2d`, and verifies that each receiver CTA receives the same tile.
它已经会 program multicast descriptor、设置 `cluster_dim[0]`、使用 `vortex::dxa_multicast_2d`，并验证每个 receiver CTA 都收到同一个 tile。

It is currently a correctness smoke test rather than the full benchmark.
它目前是 correctness smoke test，而不是完整 benchmark。

It currently chooses `num_recv = VX_CFG_NUM_WARPS`, so it is not fixed to the requested 4 CTAs.
它目前选择 `num_recv = VX_CFG_NUM_WARPS`，所以不是需求里的固定 4 CTA。

It currently requires `tile_cols == VX_CFG_NUM_THREADS`, so it cannot sweep 16, 32, 64, and 128 columns across all thread-count configurations.
它目前要求 `tile_cols == VX_CFG_NUM_THREADS`，所以无法在所有 thread-count 配置下 sweep 16、32、64、128 列。

It currently has only multicast mode, so it cannot compute multicast speedup against non-multicast per-CTA DXA inside the same launch geometry.
它目前只有 multicast mode，所以无法在相同 launch geometry 下相对于 non-multicast per-CTA DXA 计算 multicast speedup。

`tests/regression/dxa_kmajor_check` should remain a layout-specific correctness test.
`tests/regression/dxa_kmajor_check` 应保持为 layout-specific correctness test。

`tests/regression/sgemm2_dxa_mcast` and `tests/regression/sgemm_tcu_wg_dxa_mcast` should be used as application-level sanity references, not as the primary copy microbenchmark.
`tests/regression/sgemm2_dxa_mcast` 和 `tests/regression/sgemm_tcu_wg_dxa_mcast` 应作为 application-level sanity reference，而不是主要 copy microbenchmark。

---

## File Structure
## 文件结构

Modify `tests/regression/dxa_copy_mcast/common.h` to add mode constants and explicit benchmark arguments.
修改 `tests/regression/dxa_copy_mcast/common.h`，加入 mode 常量和显式 benchmark 参数。

Modify `tests/regression/dxa_copy_mcast/main.cpp` to support smoke mode, `--mode=percta|mcast`, `--num-ctas=4`, arbitrary tile shape, stable summary output, and cluster launch setup.
修改 `tests/regression/dxa_copy_mcast/main.cpp`，支持 smoke mode、`--mode=percta|mcast`、`--num-ctas=4`、任意 tile shape、稳定 summary 输出，以及 cluster launch setup。

Modify `tests/regression/dxa_copy_mcast/kernel.cpp` to share one tile-copy verification path while selecting per-CTA DXA or rank-0 multicast.
修改 `tests/regression/dxa_copy_mcast/kernel.cpp`，共享一个 tile-copy verification path，同时选择 per-CTA DXA 或 rank-0 multicast。

Modify `tests/regression/dxa_copy` in place, without forking it, so large 2D tiles use strided per-thread LSU copy and a single-warp launch geometry.
就地修改 `tests/regression/dxa_copy`，不 fork 它，让大的 2D tile 使用 per-thread stride LSU copy 和 single-warp launch geometry。

Modify `tests/regression/dxa_copy/Makefile` to preserve the default DXA build while allowing the sweep runner to force the LSU baseline.
修改 `tests/regression/dxa_copy/Makefile`，保留默认 DXA build，同时允许 sweep runner 强制选择 LSU baseline。

Create `tools/dxa/run_copy_sweep.py` to run Figure 3(b) using `dxa_copy` and Figure 3(c) using `dxa_copy_mcast`.
创建 `tools/dxa/run_copy_sweep.py`，用 `dxa_copy` 跑 Figure 3(b)，用 `dxa_copy_mcast` 跑 Figure 3(c)。

Create `tools/dxa/plot_copy_sweep.py` to render 3-by-3 hardware grids with 4-by-4 tile heatmaps.
创建 `tools/dxa/plot_copy_sweep.py`，画出 3-by-3 hardware grid，每个格子是 4-by-4 tile heatmap。

Create `docs/results/dxa_copy_sweep/README.md` to document commands, CSV columns, and poster-safe interpretation.
创建 `docs/results/dxa_copy_sweep/README.md`，记录命令、CSV 列和 poster 中安全表述。

Do not modify `tests/regression/Makefile` for a new app in this plan.
本计划不为了新 app 修改 `tests/regression/Makefile`。

---

## Task 1: Lock Existing Behavior Before Re-Engineering
## Task 1：重构前锁住现有行为

**Files:**
**文件：**

- Modify: `tests/regression/dxa_copy_mcast/main.cpp`
- 修改：`tests/regression/dxa_copy_mcast/main.cpp`

- Modify: `tests/regression/dxa_copy_mcast/kernel.cpp`
- 修改：`tests/regression/dxa_copy_mcast/kernel.cpp`

- Modify: `tests/regression/dxa_copy_mcast/common.h`
- 修改：`tests/regression/dxa_copy_mcast/common.h`

- [x] **Step 1: Record the current smoke command.**
- [x] **步骤 1：记录当前 smoke command。**

Run from `build/` after configure:
在 configure 之后从 `build/` 目录运行：

```bash
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=4 --l2cache --perf=6 --app=dxa_copy_mcast --args="-r16 -c16 -R16 -C4"
```

Expected: the app prints `PASSED`.
预期结果：app 打印 `PASSED`。

- [x] **Step 2: Add an explicit `--mode=smoke` spelling without changing defaults.**
- [x] **步骤 2：添加显式 `--mode=smoke` 写法，但不改变默认行为。**

Keep the existing default equivalent to:
保持现有默认行为等价于：

```text
mode=smoke
num_ctas=VX_CFG_NUM_WARPS
tile_cols=VX_CFG_NUM_THREADS
```

This preserves the existing correctness test.
这样可以保留现有 correctness test。

- [x] **Step 3: Verify smoke mode still passes.**
- [x] **步骤 3：验证 smoke mode 仍然通过。**

Run:
运行：

```bash
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=4 --l2cache --perf=6 --app=dxa_copy_mcast --args="--mode=smoke -r16 -c16 -R16 -C4"
```

Expected: the app prints `PASSED`.
预期结果：app 打印 `PASSED`。

---

## Task 2: Add Benchmark Modes To `dxa_copy_mcast`
## Task 2：给 `dxa_copy_mcast` 增加 benchmark modes

**Files:**
**文件：**

- Modify: `tests/regression/dxa_copy_mcast/common.h`
- 修改：`tests/regression/dxa_copy_mcast/common.h`

- Modify: `tests/regression/dxa_copy_mcast/main.cpp`
- 修改：`tests/regression/dxa_copy_mcast/main.cpp`

- Modify: `tests/regression/dxa_copy_mcast/kernel.cpp`
- 修改：`tests/regression/dxa_copy_mcast/kernel.cpp`

- [x] **Step 1: Add mode constants.**
- [x] **步骤 1：添加 mode constants。**

Use these values in `common.h`:
在 `common.h` 中使用这些值：

```cpp
enum : uint32_t {
  DXA_COPY_MCAST_MODE_SMOKE  = 0,
  DXA_COPY_MCAST_MODE_PERCTA = 1,
  DXA_COPY_MCAST_MODE_MCAST  = 2,
};
```

- [x] **Step 2: Add only kernel-required fields to `kernel_arg_t`.**
- [x] **步骤 2：只给 `kernel_arg_t` 添加 kernel 必需字段。**

Use fields that keep host and kernel self-describing without passing host-only state:
使用能让 host 和 kernel 自描述的字段，但不传 host-only 状态：

```cpp
uint32_t mode;
```

- [x] **Step 3: Add CLI parsing.**
- [x] **步骤 3：添加 CLI parsing。**

Support these flags:
支持这些参数：

```text
--mode=smoke|percta|mcast
--num-ctas=N
--verify=0|1
```

Keep `-r`, `-c`, `-R`, `-C`, and `-k` compatible with the current test.
保持 `-r`、`-c`、`-R`、`-C` 和 `-k` 与当前测试兼容。

- [x] **Step 4: Change benchmark launch geometry only for `percta` and `mcast`.**
- [x] **步骤 4：只在 `percta` 和 `mcast` 下改变 benchmark launch geometry。**

Use exactly four CTAs for Figure 3(c):
Figure 3(c) 使用固定四个 CTA：

```text
grid_dim[0] = 4
cluster_dim[0] = 4
```

Reject benchmark runs where the hardware cannot co-reside four single-warp CTAs.
如果硬件无法 co-reside 四个 single-warp CTA，就拒绝 benchmark run。

- [x] **Step 5: Remove the benchmark-only `tile_cols == num_threads` restriction.**
- [x] **步骤 5：移除 benchmark-only 场景下的 `tile_cols == num_threads` 限制。**

Keep that restriction only for `smoke` mode if it helps preserve the old test.
如果它有助于保留旧测试，就只在 `smoke` mode 中保留这个限制。

For benchmark modes, let each thread process multiple elements with a stride of `blockDim.x`.
对于 benchmark modes，让每个线程以 `blockDim.x` 为 stride 处理多个元素。

- [x] **Step 6: Implement per-CTA DXA mode.**
- [x] **步骤 6：实现 per-CTA DXA mode。**

Each CTA registers one local barrier transaction and issues its own `vx_dxa_issue_2d_wg`.
每个 CTA 注册一个 local barrier transaction，并独立 issue 自己的 `vx_dxa_issue_2d_wg`。

The per-CTA mode must use the same matrix, same tile, same local-memory size, same block size, and same verification path as multicast mode.
per-CTA mode 必须使用与 multicast mode 相同的 matrix、tile、local-memory size、block size 和 verification path。

- [x] **Step 7: Keep multicast mode on the existing helper.**
- [x] **步骤 7：multicast mode 继续使用现有 helper。**

Use `vortex::dxa_multicast_2d` so the mask, `expect_tx`, group rendezvous, and rank-0 issue logic stay centralized.
使用 `vortex::dxa_multicast_2d`，让 mask、`expect_tx`、group rendezvous 和 rank-0 issue 逻辑保持集中。

- [x] **Step 8: Add a stable summary line.**
- [x] **步骤 8：添加稳定 summary line。**

Print one machine-readable line after verification:
verification 之后打印一行 machine-readable summary：

```text
DXA_COPY_MCAST_RESULT mode=mcast rows=512 cols=512 tile_rows=16 tile_cols=16 num_ctas=4 verify=1 status=PASS
```

- [x] **Step 9: Verify both benchmark modes in SimX.**
- [x] **步骤 9：在 SimX 中验证两个 benchmark modes。**

Run:
运行：

```bash
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=6 --app=dxa_copy_mcast --args="--mode=percta --num-ctas=4 -r512 -c512 -R16 -C16 --verify=1"
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=6 --app=dxa_copy_mcast --args="--mode=mcast --num-ctas=4 -r512 -c512 -R16 -C16 --verify=1"
```

Expected: both runs print `PASSED` and `DXA_COPY_MCAST_RESULT`.
预期结果：两次运行都打印 `PASSED` 和 `DXA_COPY_MCAST_RESULT`。

- [x] **Step 10: Fix Figure 3(c) benchmark mode to cover the full matrix, not one tile.**
- [x] **步骤 10：修正 Figure 3(c) benchmark mode，使其覆盖整张 matrix，而不是单个 tile。**

Benchmark `percta` and `mcast` modes now loop over every tile in the matrix and verify `num_ctas` full-matrix output copies.
现在 benchmark 的 `percta` 和 `mcast` modes 会遍历 matrix 中的所有 tile，并验证 `num_ctas` 份完整 matrix 输出。

The old `simx_full.csv` 3(c) rows were collected before this correction and should not be used for poster plots.
旧的 `simx_full.csv` 中的 3(c) 行是在这个修正之前采集的，不应再用于 poster 作图。

Verified evidence:
已验证证据：

```text
docs/results/dxa_copy_sweep/simx_3c_fullmatrix.csv
```

For `NW=8`, `NT=8`, `tile=16x16`, the full-matrix 512x512 SimX counter ratio is `percta dxa_gmem_reads=65536` versus `mcast dxa_gmem_reads=16384`, while both have `dxa_lmem_writes=65536`.
对于 `NW=8`、`NT=8`、`tile=16x16`，full-matrix 512x512 SimX counter ratio 是 `percta dxa_gmem_reads=65536` 对比 `mcast dxa_gmem_reads=16384`，同时两者都是 `dxa_lmem_writes=65536`。

---

## Task 3: Reuse `dxa_copy` For Figure 3(b)
## Task 3：复用 `dxa_copy` 跑 Figure 3(b)

**Files:**
**文件：**

- Read: `tests/regression/dxa_copy/main.cpp`
- 读取：`tests/regression/dxa_copy/main.cpp`

- Read: `tests/regression/dxa_copy/kernel.cpp`
- 读取：`tests/regression/dxa_copy/kernel.cpp`

- Modify: `tests/regression/dxa_copy/main.cpp`
- 修改：`tests/regression/dxa_copy/main.cpp`

- Modify: `tests/regression/dxa_copy/kernel.cpp`
- 修改：`tests/regression/dxa_copy/kernel.cpp`

- Modify: `tests/regression/dxa_copy/Makefile`
- 修改：`tests/regression/dxa_copy/Makefile`

- Create: `tools/dxa/run_copy_sweep.py`
- 创建：`tools/dxa/run_copy_sweep.py`

- [x] **Step 1: Confirm the existing command shape.**
- [x] **步骤 1：确认现有命令形态。**

Use `dxa_copy` directly for matrix 512-by-512:
直接用 `dxa_copy` 跑 512-by-512 matrix：

```bash
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=6 --app=dxa_copy --args="-d2 -s0 512 -s1 512 -t0 16 -t1 16"
```

The DXA-vs-LSU comparison must come from matched builds with and without `VX_CFG_EXT_DXA_ENABLE`.
DXA-vs-LSU 对比必须来自启用和不启用 `VX_CFG_EXT_DXA_ENABLE` 的 matched builds。

- [x] **Step 2: Add no new Figure 3(b) kernel unless this command cannot be made reproducible.**
- [x] **步骤 2：除非这个命令无法复现，否则不要为 Figure 3(b) 新增 kernel。**

If a new helper is needed, add it to the runner rather than duplicating `dxa_copy`.
如果需要新 helper，把它加到 runner，而不是复制 `dxa_copy`。

The implemented path keeps `dxa_copy` as the app and only re-engineers its block geometry, LSU loop, and DXA/LSU Makefile selection.
已实现路径仍然保留 `dxa_copy` 这个 app，只重构它的 block geometry、LSU loop 和 DXA/LSU Makefile 选择。

---

## Task 4: Add A Unified Sweep Runner
## Task 4：添加统一 sweep runner

**Files:**
**文件：**

- Create: `tools/dxa/run_copy_sweep.py`
- 创建：`tools/dxa/run_copy_sweep.py`

- [x] **Step 1: Implement the hardware matrix.**
- [x] **步骤 1：实现 hardware matrix。**

Use these hardware configurations:
使用这些硬件配置：

```python
WARPS = [8, 16, 32]
THREADS = [8, 16, 32]
TILES = [(r, c) for r in [16, 32, 64, 128] for c in [16, 32, 64, 128]]
```

- [x] **Step 2: Implement Figure 3(b) cases.**
- [x] **步骤 2：实现 Figure 3(b) cases。**

For each hardware configuration and tile, run LSU and DXA variants of `dxa_copy`.
对每个 hardware configuration 和 tile，运行 `dxa_copy` 的 LSU 和 DXA variants。

Record one CSV row per variant.
每个 variant 记录一行 CSV。

- [x] **Step 3: Implement Figure 3(c) cases.**
- [x] **步骤 3：实现 Figure 3(c) cases。**

For each hardware configuration and tile, run `dxa_copy_mcast --mode=percta` and `dxa_copy_mcast --mode=mcast`.
对每个 hardware configuration 和 tile，运行 `dxa_copy_mcast --mode=percta` 和 `dxa_copy_mcast --mode=mcast`。

Record one CSV row per variant.
每个 variant 记录一行 CSV。

- [x] **Step 4: Preserve timeout rows instead of fabricating data.**
- [x] **步骤 4：保留 timeout rows，而不是伪造数据。**

If RTLsim exceeds the chosen timeout, write `status=TIMEOUT` and leave numeric result columns empty.
如果 RTLsim 超过选定 timeout，写入 `status=TIMEOUT`，并让数值结果列为空。

- [x] **Step 5: Use generated build scripts from `build/`.**
- [x] **步骤 5：从 `build/` 使用生成的 build scripts。**

The runner must call `./ci/blackbox.sh` from the generated build directory.
runner 必须从生成的 build 目录调用 `./ci/blackbox.sh`。

---

## Task 5: Add Plotting For Poster Figures
## Task 5：添加 poster 图表绘制

**Files:**
**文件：**

- Create: `tools/dxa/plot_copy_sweep.py`
- 创建：`tools/dxa/plot_copy_sweep.py`

- [x] **Step 1: Plot Figure 3(b) speedup.**
- [x] **步骤 1：绘制 Figure 3(b) speedup。**

Compute `speedup = cycles_lsu / cycles_dxa`.
计算 `speedup = cycles_lsu / cycles_dxa`。

Render a 3-by-3 hardware grid.
渲染 3-by-3 hardware grid。

Each subplot is a 4-by-4 heatmap over tile rows and tile columns.
每个 subplot 是 tile rows 与 tile columns 上的 4-by-4 heatmap。

- [x] **Step 2: Plot Figure 3(c) speedup.**
- [x] **步骤 2：绘制 Figure 3(c) speedup。**

Compute `speedup = cycles_percta / cycles_mcast`.
计算 `speedup = cycles_percta / cycles_mcast`。

Use the same plot layout as Figure 3(b).
使用与 Figure 3(b) 相同的图布局。

- [x] **Step 3: Mark timeout or failed cells explicitly.**
- [x] **步骤 3：显式标出 timeout 或 failed cells。**

Do not interpolate missing data.
不要插值补齐缺失数据。

---

## Task 6: RTLsim Validation And Debug Escalation
## Task 6：RTLsim 验证与 debug 升级

- [x] **Step 1: Run one RTLsim smoke pair before the full sweep.**
- [x] **步骤 1：完整 sweep 前先跑一对 RTLsim smoke。**

Run:
运行：

```bash
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=8 --threads=8 --l2cache --perf=16 --app=dxa_copy_mcast --args="--mode=percta --num-ctas=4 -r512 -c512 -R16 -C16 --verify=1"
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=8 --threads=8 --l2cache --perf=16 --app=dxa_copy_mcast --args="--mode=mcast --num-ctas=4 -r512 -c512 -R16 -C16 --verify=1"
```

Expected: both runs either pass or provide enough trace evidence for a concrete RTL investigation.
预期结果：两次运行要么通过，要么提供足够 trace evidence 供具体 RTL investigation 使用。

Observed L2-on RTLsim smoke results: per-CTA passed at 5,900,128 cycles; multicast passed at 6,245,900 cycles.
观察到的 L2-on RTLsim smoke 结果：per-CTA 在 5,900,128 cycles 通过；multicast 在 6,245,900 cycles 通过。

Re-verified after regenerating the build tree with `--tooldir=/home/chengxuan99/tools`: `docs/results/dxa_copy_sweep/rtlsim_3c_smoke_l2_fresh.csv` has 2 PASS rows with the same cycle counts.
在用 `--tooldir=/home/chengxuan99/tools` 重新生成 build tree 后已复验：`docs/results/dxa_copy_sweep/rtlsim_3c_smoke_l2_fresh.csv` 有 2 行 PASS，cycle 数与上述结果一致。

- [x] **Step 2: Escalate debug only after a long runtime or first failed smoke.**
- [x] **步骤 2：只有在长时间运行或第一次 smoke 失败后才升级 debug。**

Enable `--debug=1`, `DBG_TRACE_DXA`, and the existing DXA pipeline trace points when needed.
需要时启用 `--debug=1`、`DBG_TRACE_DXA` 和现有 DXA pipeline trace points。

Do not classify a slow RTLsim case as deadlock without trace evidence.
没有 trace evidence 时，不要把慢 RTLsim case 判定为 deadlock。

No debug escalation was needed for the L2-on RTLsim smoke pair because both commands passed within the 120-second cap.
这组 L2-on RTLsim smoke pair 都在 120 秒上限内通过，因此不需要升级 debug。

The transient return-code-2 failure seen during re-verification was a stale generated build-tree tool path (`/root/tools/verilator`), not a DXA/barrier RTL failure.
复验过程中短暂出现的 return-code-2 失败是生成 build tree 中 stale tool path（`/root/tools/verilator`）导致的，不是 DXA/barrier RTL 失败。

---

## Poster Interpretation Rules
## Poster 表述规则

Say that current Vortex DXA multicast is intra-core CTA multicast.
要说当前 Vortex DXA multicast 是 intra-core CTA multicast。

Do not claim cross-core multicast.
不要声称 cross-core multicast。

Do not claim DSM or SM-cluster support.
不要声称支持 DSM 或 SM-cluster。

Say that multicast reduces duplicate GMEM reads and DXA transactions across four co-resident CTAs.
可以说 multicast 减少了四个 co-resident CTA 之间重复的 GMEM reads 和 DXA transactions。

Do not describe current RTL as a one-beat LMEM broadcast.
不要把当前 RTL 描述成 one-beat LMEM broadcast。

Say that receiver LMEM writes are replayed serially in the current implementation.
要说当前实现中 receiver LMEM writes 是 serial replay。

---

## Verification Checklist
## 验证清单

- [x] Existing `dxa_copy_mcast` smoke behavior still passes.
- [x] 现有 `dxa_copy_mcast` smoke 行为仍然通过。

- [x] `dxa_copy_mcast --mode=percta --num-ctas=4` passes in SimX.
- [x] `dxa_copy_mcast --mode=percta --num-ctas=4` 在 SimX 中通过。

- [x] `dxa_copy_mcast --mode=mcast --num-ctas=4` passes in SimX.
- [x] `dxa_copy_mcast --mode=mcast --num-ctas=4` 在 SimX 中通过。

- [x] One RTLsim per-CTA smoke passes or produces actionable trace evidence.
- [x] 一个 RTLsim per-CTA smoke 通过，或产生可行动的 trace evidence。

- [x] One RTLsim multicast smoke passes or produces actionable trace evidence.
- [x] 一个 RTLsim multicast smoke 通过，或产生可行动的 trace evidence。

- [x] Figure 3(b) SimX sweep produces 288 variant rows before speedup aggregation.
- [x] Figure 3(b) SimX sweep 在 speedup aggregation 前产生 288 行 variant 数据。

Observed status: `docs/results/dxa_copy_sweep/simx_3b_fullmatrix.csv` has 288 PASS rows and 0 TIMEOUT rows.
观察状态：`docs/results/dxa_copy_sweep/simx_3b_fullmatrix.csv` 有 288 行 PASS 和 0 行 TIMEOUT。

The formerly failing `dxa`, `warps=32`, `threads=32`, `tile_rows=16`, `tile_cols=64` row now passes after fixing an L2 MSHR replay-capacity deadlock; the pre-fix 300-second diagnostic is retained as root-cause evidence.
之前失败的 `dxa`、`warps=32`、`threads=32`、`tile_rows=16`、`tile_cols=64` 行在修复 L2 MSHR replay-capacity deadlock 后已经通过；修复前的 300 秒诊断作为 root-cause 证据保留。

- [x] Figure 3(c) SimX sweep produces 288 variant rows before speedup aggregation.
- [x] Figure 3(c) SimX sweep 在 speedup aggregation 前产生 288 行 variant 数据。

Observed status: `docs/results/dxa_copy_sweep/simx_3c_fullmatrix.csv` has 288 PASS rows.
观察状态：`docs/results/dxa_copy_sweep/simx_3c_fullmatrix.csv` 有 288 行 PASS。

- [x] Plotter marks failures and timeouts without fabricating cells.
- [x] plotter 会标出 failures 和 timeouts，不会伪造 cell。

- [x] Poster text says intra-core CTA multicast, not cross-core multicast.
- [x] poster 文案说 intra-core CTA multicast，而不是 cross-core multicast。

## Self-Review
## 自检

Spec coverage is complete for Figure 3(b), Figure 3(c), SimX, RTLsim, 4-CTA multicast scope, and poster-safe wording.
对于 Figure 3(b)、Figure 3(c)、SimX、RTLsim、4-CTA multicast 范围和 poster-safe wording，本计划覆盖完整。

The plan removes the previous duplicate-app default and uses existing DXA tests as first-class assets.
本计划移除了之前默认新增 app 的路线，并把现有 DXA tests 当作一等复用资产。

No placeholder benchmark app remains in the plan.
计划中不再保留 placeholder benchmark app。

The main implementation risk is preserving `dxa_copy_mcast` smoke semantics while adding benchmark modes.
主要实现风险是添加 benchmark modes 的同时保持 `dxa_copy_mcast` smoke semantics。

The mitigation is to keep `--mode=smoke` equivalent to the current default and verify it before changing benchmark paths.
缓解方法是让 `--mode=smoke` 等价于当前默认行为，并在修改 benchmark paths 前先验证它。
