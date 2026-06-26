# DXA Multicast Speedup Closure And Shared-Memory Atomic Barrier Plan
# DXA Multicast Speedup 闭环与 Shared-Memory Atomic Barrier 计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.
> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，按任务逐项执行本计划。

> **Tracking rule:** Steps use checkbox (`- [ ]`) syntax and should be checked only after local verification succeeds.
> **跟踪规则：** 每一步使用 checkbox (`- [ ]`) 语法，只有本地验证通过后才勾选。

> **L2 rule:** All DXA performance and correctness runs in this plan must keep L2 enabled.
> **L2 规则：** 本计划中的所有 DXA 性能与正确性实验都必须启用 L2。

## Captured User Prompt
## 用户 Prompt 记录

Install the `fireworks-tech-graph` skill from `https://github.com/yizhiyanhua-ai/fireworks-tech-graph`.
安装来自 `https://github.com/yizhiyanhua-ai/fireworks-tech-graph` 的 `fireworks-tech-graph` skill。

The skill has been installed at `/root/.codex/skills/fireworks-tech-graph`; restart Codex to auto-load it as an available skill.
该 skill 已安装在 `/root/.codex/skills/fireworks-tech-graph`；重启 Codex 后会自动进入可用 skill 列表。

The current multicast result is not acceptable if it has no speedup, because per-CTA cannot be explained away as simply benefiting from L2 hits.
如果当前 multicast 没有 speedup，这个结果不能接受，因为不能简单解释成 per-CTA 只是受益于 L2 hit。

The user model is that multicast should roughly pay one miss-side transfer plus replay, while per-CTA should pay one miss-side transfer plus repeated hit-side transfers for the remaining CTAs.
用户的模型是 multicast 大致应付出一次 miss-side transfer 加 replay，而 per-CTA 应付出一次 miss-side transfer 再加剩余 CTA 的重复 hit-side transfer。

With one DXA worker, DXA work is serialized and additive, so the observed lack of speedup likely indicates a benchmark, datapath, synchronization, or replay problem.
在只有一个 DXA worker 的情况下，DXA 工作是串行且可加的，所以观察不到 speedup 很可能意味着 benchmark、datapath、同步或 replay 存在问题。

Between the multicast performance discussion and the correctness discussion, the real requirement is to thoroughly solve multicast having no speedup.
在 multicast 性能讨论和 correctness 讨论之间，真正的要求是彻底解决 multicast 没有 speedup 的问题。

After removing `group_sync`, correctness must be verified instead of assumed.
移除 `group_sync` 之后，正确性必须验证，不能默认成立。

Implement a Nvidia-like shared-memory plus atomic software barrier, while keeping the kernel-facing programming style as close as possible to the existing hard barrier API.
实现一个类似 Nvidia 的 shared-memory plus atomic software barrier，同时尽量保持 kernel 侧编程风格接近现有 hard barrier API。

The soft barrier should allow a caller to claim or construct a barrier, call `expect_tx` before issuing DXA, pass a soft-barrier pointer or tag into the DXA issue path, and let `arrive_wait` spin until both pending events are complete and all participating warps have arrived.
soft barrier 应允许调用方 claim 或构造 barrier，在 issue DXA 前调用 `expect_tx`，把 soft-barrier pointer 或 tag 传给 DXA issue 路径，并让 `arrive_wait` 一直等待到 pending event 完成且所有参与 warp 到齐。

The user will provide final experiment details for the shared-memory atomic barrier later, but implementation planning and enabling work should start now.
用户之后会补充 shared-memory atomic barrier 的最终实验细节，但实现规划和前置 enabling work 现在就要开始。

The multicast bottleneck should be fixed architecturally by moving replay out of the DXA worker drain path and into logic near the SMEM/LMEM bank side.
multicast 瓶颈应通过架构方式修复：把 replay 从 DXA worker drain path 移出，放到靠近 SMEM/LMEM bank 侧的逻辑中。

The desired multicast packet is one data packet plus the first CTA SMEM address, byte offset or byte enable, and multicast replay metadata such as CTA mask, replay count, or stride.
期望的 multicast packet 是一个 data packet 加首个 CTA 的 SMEM 地址、byte offset 或 byte enable，以及 CTA mask、replay count 或 stride 等 multicast replay metadata。

The hard-versus-soft barrier comparison should use absolute latency deviation and overhead ratio across workload sizes.
hard-versus-soft barrier 对比应使用绝对延迟偏差和不同 workload size 下的 overhead ratio。

The delivery target is poster-quality Figure 3(c) multicast speedup data plus hard/soft barrier plots that show nearly constant hard-barrier overhead and much higher soft-barrier overhead for small shared-memory workloads.
交付目标是 poster-quality 的 Figure 3(c) multicast speedup 数据图，以及 hard/soft barrier 图，展示 hard barrier overhead 几乎恒定、而 soft barrier 在小 shared-memory workload 下 overhead 明显更高。

## Current Evidence
## 当前证据

`tests/regression/dxa_copy_mcast` already has reusable host and kernel code for intra-core CTA multicast, so it remains the primary Figure 3(c) benchmark vehicle.
`tests/regression/dxa_copy_mcast` 已经有可复用的 intra-core CTA multicast host 和 kernel 代码，所以继续作为 Figure 3(c) 的主要 benchmark 载体。

The current no-`group_sync` path has passed basic SimX sample correctness and a small full-writeback correctness case, but it has not yet passed a forced early-completion stress where the issuer completes before receiver CTAs call `expect_tx`.
当前去掉 `group_sync` 的路径已经通过基本 SimX sample correctness 和一个小规模 full-writeback correctness case，但还没有通过强制 issuer 在 receiver CTA 调用 `expect_tx` 之前完成的 early-completion stress。

Observed on 2026-06-25: `dxa_copy_mcast --mode=mcast --writeback=sample --num-ctas=4 -r 512 -c 512 -R 16 -C 16 --verify=1` passed in SimX with L2 enabled at 780462 cycles.
2026-06-25 观察结果：`dxa_copy_mcast --mode=mcast --writeback=sample --num-ctas=4 -r 512 -c 512 -R 16 -C 16 --verify=1` 在 L2 enabled 的 SimX 中通过，cycles 为 780462。

Observed on 2026-06-25: `dxa_copy_mcast --mode=mcast --writeback=full --num-ctas=4 -r 64 -c 64 -R 16 -C 16 --verify=1` passed in SimX with L2 enabled at 169786 cycles.
2026-06-25 观察结果：`dxa_copy_mcast --mode=mcast --writeback=full --num-ctas=4 -r 64 -c 64 -R 16 -C 16 --verify=1` 在 L2 enabled 的 SimX 中通过，cycles 为 169786。

`sw/kernel/include/vx_dxa.h` currently packs only `barrier_id` and `desc_slot` into DXA metadata through `vx_dxa_pack_meta(desc_slot, barrier_id)`.
`sw/kernel/include/vx_dxa.h` 当前通过 `vx_dxa_pack_meta(desc_slot, barrier_id)` 只把 `barrier_id` 和 `desc_slot` 打包进 DXA metadata。

The existing metadata format cannot carry a full shared-memory pointer without either a new issue encoding or an additional sideband path.
现有 metadata 格式无法携带完整 shared-memory pointer，除非增加新的 issue encoding 或额外 sideband path。

SimX shared/local-memory AMO support is now implemented for the operations needed by the first soft-barrier protocol.
SimX 的 shared/local-memory AMO support 现在已经实现，可覆盖第一版 soft-barrier protocol 需要的操作。

RTL local-memory AMO support is now implemented by preserving AMO sideband through `hw/rtl/mem/VX_lmem_switch.sv` and performing per-bank RMW in `hw/rtl/mem/VX_local_mem.sv`.
RTL 的 local-memory AMO support 现在已经实现：`hw/rtl/mem/VX_lmem_switch.sv` 保留 AMO sideband，并在 `hw/rtl/mem/VX_local_mem.sv` 中执行 per-bank RMW。

The verified local-memory AMO subset includes AMOADD, AMOSWAP, AMOOR, AMOAND, AMOXOR, AMOMAX, AMOMINU, and AMOADD.AQRL in both SimX and RTLsim.
已验证的 local-memory AMO 子集包括 AMOADD、AMOSWAP、AMOOR、AMOAND、AMOXOR、AMOMAX、AMOMINU 和 AMOADD.AQRL，并且 SimX 与 RTLsim 都通过。

LR/SC local-memory mode is not required for the first phase-counter soft barrier and currently needs separate investigation because the larger local LR/SC regression timed out at the 120s cap.
第一版 phase-counter soft barrier 不需要 LR/SC local-memory mode；较大的 local LR/SC regression 当前在 120s timeout cap 下超时，因此作为独立问题后续调查。

Observed on 2026-06-25: `smem_atomic_barrier -b1024 -i4` passed in SimX with L2 enabled at `pending=0`, `phase=4`, `arrived=16`, `register_cycles=400`, `event_cycles=7433`, and `release_cycles=9188`.
2026-06-25 观察结果：`smem_atomic_barrier -b1024 -i4` 在 L2 enabled 的 SimX 中通过，结果为 `pending=0`、`phase=4`、`arrived=16`、`register_cycles=400`、`event_cycles=7433` 和 `release_cycles=9188`。

Observed on 2026-06-25: `smem_atomic_barrier -b1024 -i4` passed in RTLsim with L2 enabled at `pending=0`, `phase=4`, `arrived=16`, `register_cycles=365`, `event_cycles=7432`, and `release_cycles=9059`.
2026-06-25 观察结果：`smem_atomic_barrier -b1024 -i4` 在 L2 enabled 的 RTLsim 中通过，结果为 `pending=0`、`phase=4`、`arrived=16`、`register_cycles=365`、`event_cycles=7432` 和 `release_cycles=9059`。

The C1 software-completion soft-barrier SimX sweep is stored at `docs/results/dxa_barrier_overhead/simx_soft_smem_barrier.csv`, with plot `docs/results/dxa_barrier_overhead/soft_smem_barrier_overhead.png`.
C1 software-completion soft-barrier 的 SimX sweep 存在 `docs/results/dxa_barrier_overhead/simx_soft_smem_barrier.csv`，对应图为 `docs/results/dxa_barrier_overhead/soft_smem_barrier_overhead.png`。

That SimX sweep has 6/6 PASS rows; per-iteration software barrier overhead is 438.75 cycles across 1KB, 2KB, 4KB, 8KB, 16KB, and 32KB payloads, while overhead ratio drops from 19.10% to 0.80%.
该 SimX sweep 有 6/6 个 PASS rows；在 1KB、2KB、4KB、8KB、16KB 和 32KB payload 上，per-iteration software barrier overhead 均为 438.75 cycles，而 overhead ratio 从 19.10% 降至 0.80%。

The matching RTLsim C1 sweep is stored at `docs/results/dxa_barrier_overhead/rtlsim_soft_smem_barrier.csv`, with plot under `docs/results/dxa_barrier_overhead/rtlsim_soft_plots/`.
匹配的 RTLsim C1 sweep 存在 `docs/results/dxa_barrier_overhead/rtlsim_soft_smem_barrier.csv`，对应图在 `docs/results/dxa_barrier_overhead/rtlsim_soft_plots/` 下。

That RTLsim sweep has 6/6 PASS rows; per-iteration software barrier overhead is about 407 to 421 cycles, while overhead ratio drops from 17.96% to 0.77%.
该 RTLsim sweep 有 6/6 个 PASS rows；per-iteration software barrier overhead 约为 407 到 421 cycles，而 overhead ratio 从 17.96% 降至 0.77%。

Observed on 2026-06-25: `dxa_copy -d2 -s0 32 -s1 32 -t0 4 -t1 4 -S` initially failed in SimX by decoding the soft tag as hard barrier release `event_release(1024)`, then passed after adding raw soft-barrier metadata and LMEM event decrement.
2026-06-25 观察结果：`dxa_copy -d2 -s0 32 -s1 32 -t0 4 -t1 4 -S` 最初在 SimX 中把 soft tag 解成 hard barrier release `event_release(1024)` 而失败；加入 raw soft-barrier metadata 和 LMEM event decrement 后通过。

Observed on 2026-06-25: the same `dxa_copy -S` case initially timed out in RTLsim, then passed after widening the DXA LMEM completion attr to carry the raw 27-bit barrier payload and emitting a near-LMEM AMOADD(-1) micro-op for soft completions.
2026-06-25 观察结果：同一个 `dxa_copy -S` case 最初在 RTLsim 中 timeout；将 DXA LMEM completion attr 扩展为携带 raw 27-bit barrier payload，并为 soft completion 发出 near-LMEM AMOADD(-1) micro-op 后通过。

The matching hard-barrier `dxa_copy` smoke still passes in SimX and RTLsim after the C2 change, so the hard completion path remains intact.
C2 修改后，匹配的 hard-barrier `dxa_copy` smoke 仍在 SimX 和 RTLsim 中通过，因此 hard completion path 未被破坏。

The C1 `smem_atomic_barrier -b1024 -i4` smoke still passes in SimX and RTLsim after the C2 change, so the existing local-memory AMO path remains intact.
C2 修改后，C1 `smem_atomic_barrier -b1024 -i4` smoke 仍在 SimX 和 RTLsim 中通过，因此已有 local-memory AMO path 未被破坏。

Observed on 2026-06-25: `dxa_copy -B` passed the hard DXA completion marker in RTLsim at 1KB with `release_cycles=831`, and `dxa_copy -B -S` passed the soft DXA completion marker at `release_cycles=1289`.
2026-06-25 观察结果：`dxa_copy -B` 在 RTLsim 的 1KB hard DXA completion marker 中通过，`release_cycles=831`；`dxa_copy -B -S` 在 soft DXA completion marker 中通过，`release_cycles=1289`。

The C2 DXA-completion SimX sweep is stored at `docs/results/dxa_barrier_overhead/simx_dxa_barrier_latency.csv`, with plot `docs/results/dxa_barrier_overhead/simx_dxa_barrier_latency.png`.
C2 DXA-completion 的 SimX sweep 存在 `docs/results/dxa_barrier_overhead/simx_dxa_barrier_latency.csv`，对应图为 `docs/results/dxa_barrier_overhead/simx_dxa_barrier_latency.png`。

That SimX sweep has 12/12 PASS rows across hard and soft DXA completion; soft release latency exceeds hard by 496 cycles at 1KB and 257 cycles at 32KB, while the extra-release ratio drops from 55.73% to 5.44%.
该 SimX sweep 在 hard 和 soft DXA completion 上共有 12/12 个 PASS rows；soft release latency 在 1KB 时比 hard 多 496 cycles，在 32KB 时多 257 cycles，而 extra-release ratio 从 55.73% 降到 5.44%。

The matching RTLsim C2 DXA-completion sweep is stored at `docs/results/dxa_barrier_overhead/rtlsim_dxa_barrier_latency.csv`, with plot `docs/results/dxa_barrier_overhead/rtlsim_dxa_barrier_latency.png`.
匹配的 RTLsim C2 DXA-completion sweep 存在 `docs/results/dxa_barrier_overhead/rtlsim_dxa_barrier_latency.csv`，对应图为 `docs/results/dxa_barrier_overhead/rtlsim_dxa_barrier_latency.png`。

That RTLsim sweep has 12/12 PASS rows; soft release latency exceeds hard by 483 cycles at 1KB and 250 cycles at 32KB, while the extra-release ratio drops from 54.21% to 5.35%.
该 RTLsim sweep 有 12/12 个 PASS rows；soft release latency 在 1KB 时比 hard 多 483 cycles，在 32KB 时多 250 cycles，而 extra-release ratio 从 54.21% 降到 5.35%。

These C2 numbers are kernel-visible release-latency deltas from `expect_tx` to barrier release, not yet a completion-detector timestamp for the exact DXA event-finish cycle.
这些 C2 数字是从 `expect_tx` 到 barrier release 的 kernel-visible release-latency deltas，还不是 completion-detector 内部记录的精确 DXA event-finish cycle timestamp。

SimX now has a prototype of the requested near-SMEM multicast replay path: DXA emits one LMEM packet with `dxa_mcast_count` and `dxa_mcast_stride`, and `sim/simx/mem/local_mem.cpp` performs receiver replay near local memory.
SimX 现在已有用户要求的 near-SMEM multicast replay prototype：DXA 发出一个带 `dxa_mcast_count` 和 `dxa_mcast_stride` 的 LMEM packet，由 `sim/simx/mem/local_mem.cpp` 在 local memory 近端执行 receiver replay。

SimX completion now releases one hard-barrier event per receiver CTA when a multicast last packet completes, using the contiguous barrier-id encoding.
SimX completion 现在会在 multicast last packet 完成时，利用连续 barrier-id encoding 为每个 receiver CTA release 一个 hard-barrier event。

With the near-SMEM replay prototype, `dxa_copy_mcast` 16x16 sample correctness passes with L2 enabled, and reported DXA LMEM write packets drop from 65536 to 16384 for the 4-CTA multicast case.
使用 near-SMEM replay prototype 后，`dxa_copy_mcast` 16x16 sample correctness 在 L2 enabled 下通过，并且 4-CTA multicast case 报告的 DXA LMEM write packets 从 65536 降到 16384。

The same 16x16 case still does not show speedup after replay offload, so receiver replay traffic alone is not the whole bottleneck for very small tiles.
同一个 16x16 case 在 replay offload 后仍然没有 speedup，因此 receiver replay traffic 本身不是 very small tile 的完整瓶颈。

A 128x128 sample case with L2 enabled and `VX_CFG_LMEM_LOG_SIZE=18` shows multicast benefit after the SimX near-SMEM replay change: per-CTA is 225560 cycles and multicast is 137571 cycles, or about 1.64x speedup.
在 L2 enabled 且 `VX_CFG_LMEM_LOG_SIZE=18` 下，128x128 sample case 在 SimX near-SMEM replay 修改后已经显示 multicast benefit：per-CTA 为 225560 cycles，multicast 为 137571 cycles，约 1.64x speedup。

The current SimX W8/T8 512x512 probe is stored at `docs/results/dxa_copy_sweep/simx_3c_smem_replay_probe_w8t8.csv` and its plot is under `docs/results/dxa_copy_sweep/simx_3c_smem_replay_probe_w8t8_plots/`.
当前 SimX W8/T8 512x512 probe 存在 `docs/results/dxa_copy_sweep/simx_3c_smem_replay_probe_w8t8.csv`，对应图在 `docs/results/dxa_copy_sweep/simx_3c_smem_replay_probe_w8t8_plots/`。

That SimX probe has 32/32 PASS rows; 10 of 16 tile cases have speedup above 1.0, with speedup ranging from 0.766 to 1.640.
该 SimX probe 有 32/32 个 PASS rows；16 个 tile cases 中有 10 个 speedup 大于 1.0，speedup 范围为 0.766 到 1.640。

Observed on 2026-06-25: the complete SimX near-SMEM replay Figure 3(c) sweep is stored at `docs/results/dxa_copy_sweep/simx_3c_smem_replay.csv`, with plot `docs/results/dxa_copy_sweep/simx_3c_smem_replay_plots/figure3c_speedup.png`.
2026-06-25 观察结果：完整 SimX near-SMEM replay Figure 3(c) sweep 存在 `docs/results/dxa_copy_sweep/simx_3c_smem_replay.csv`，对应图为 `docs/results/dxa_copy_sweep/simx_3c_smem_replay_plots/figure3c_speedup.png`。

That SimX Figure 3(c) sweep has 288/288 PASS rows and 144 paired speedups; speedup ranges from 0.765x to 1.645x with an average of 1.039x.
该 SimX Figure 3(c) sweep 有 288/288 个 PASS rows 和 144 个 paired speedups；speedup 范围为 0.765x 到 1.645x，平均为 1.039x。

The best SimX cases are large tiles such as 128x128 at NW=8, NT=8 with 1.645x speedup, while the 16x16 small-tile cases remain slower than per-CTA because fixed multicast replay and barrier/scheduling overhead dominates.
最好的 SimX case 是 128x128 这类大 tile，例如 NW=8、NT=8 时 speedup 为 1.645x；而 16x16 小 tile 仍慢于 per-CTA，因为固定 multicast replay 与 barrier/scheduling overhead 占主导。

The SimX multicast counters show the intended GMEM traffic reduction: per-CTA uses 65536 DXA GMEM reads while multicast uses 16384 reads for the 512x512, 4-CTA cases.
SimX multicast counters 显示了预期的 GMEM traffic reduction：在 512x512、4-CTA cases 中，per-CTA 使用 65536 次 DXA GMEM reads，而 multicast 使用 16384 次 reads。

Increasing multicast participants from 4 CTAs to 8 CTAs does not fix the 16x16 small-tile case; the current W8/T8 8-CTA sample probe gives per-CTA 617751 cycles and multicast 936238 cycles.
把 multicast 参与者从 4 个 CTA 增加到 8 个 CTA 不能修复 16x16 small-tile case；当前 W8/T8 8-CTA sample probe 中 per-CTA 为 617751 cycles，multicast 为 936238 cycles。

The generic software `pipeline_depth=2` experiment is a negative result for 16x16: correctness passes, but mcast sample grows to 1234618 cycles after inlining and is therefore not a poster path.
通用软件 `pipeline_depth=2` 实验对 16x16 是负结果：正确性通过，但 inline 后 mcast sample 增长到 1234618 cycles，因此不是 poster 路径。

RTLsim now has the same near-LMEM replay architecture: `VX_dxa_smem_wr` emits one base packet with multicast metadata, and `VX_dxa_lmem_mcast_replay` expands it before the LMEM DMA arbiter.
RTLsim 现在也有同样的 near-LMEM replay 架构：`VX_dxa_smem_wr` 发出一个带 multicast metadata 的 base packet，由 `VX_dxa_lmem_mcast_replay` 在 LMEM DMA arbiter 前展开。

The RTL 64x64 matrix, 16x16 tile, 4-CTA multicast smoke passes and reports `lmem_writes=256`, proving the old worker-local four-receiver replay is no longer on the RTL performance path for that case.
RTL 64x64 matrix、16x16 tile、4-CTA multicast smoke 通过，并报告 `lmem_writes=256`，证明该 case 下旧的 worker-local four-receiver replay 已经不在 RTL performance path 上。

The RTL representative W8/T8 128x128 matrix probe is stored at `docs/results/dxa_copy_sweep/rtlsim_3c_smem_replay_w8t8_128m_tiles16_32_64.csv` and its plot is under `docs/results/dxa_copy_sweep/rtlsim_3c_smem_replay_w8t8_128m_tiles16_32_64_plots/`.
RTL representative W8/T8 128x128 matrix probe 存在 `docs/results/dxa_copy_sweep/rtlsim_3c_smem_replay_w8t8_128m_tiles16_32_64.csv`，对应图在 `docs/results/dxa_copy_sweep/rtlsim_3c_smem_replay_w8t8_128m_tiles16_32_64_plots/`。

That RTL representative probe has 18/18 PASS rows; only the 64x64 tile is clearly above 1.0 speedup at 1.071, so RTL agrees with SimX that small tiles are dominated by per-tile synchronization and scheduling overhead after replay offload.
该 RTL representative probe 有 18/18 个 PASS rows；只有 64x64 tile 明确超过 1.0 speedup，达到 1.071，因此 RTL 与 SimX 一致表明 replay offload 后 small tiles 仍由 per-tile synchronization 和 scheduling overhead 主导。

The current working hypothesis is that small-tile multicast underfills the DXA/GMEM pipeline because each CTA cluster issues one short multicast transfer and then all CTAs wait, while per-CTA has more independent transfer work that keeps the single DXA worker and GMEM request pipeline occupied.
当前 working hypothesis 是 small-tile multicast 让 DXA/GMEM pipeline underfill：每个 CTA cluster 只 issue 一个很短的 multicast transfer 然后所有 CTA 都等待，而 per-CTA 有更多独立 transfer work 可以让单个 DXA worker 和 GMEM request pipeline 保持忙碌。

Therefore the next correctness-preserving performance fix should evaluate software pipelining or transfer aggregation, not only the already-offloaded receiver replay path.
因此下一步保持 correctness 的性能修复应评估 software pipelining 或 transfer aggregation，而不仅是已经 offload 的 receiver replay path。

Before the replay-offload fix, RTL multicast replay in `hw/rtl/dxa/VX_dxa_smem_wr.sv` kept `smem_wr_ready_internal` low for multicast until the last receiver replay beat of the current word.
在 replay-offload 修复前，RTL multicast replay 在 `hw/rtl/dxa/VX_dxa_smem_wr.sv` 中会让 `smem_wr_ready_internal` 在 multicast 当前 word 的最后一个 receiver replay beat 之前保持不前进。

That old RTL behavior meant one GMEM response word could hold the DXA drain path while receiver LMEM writes were replayed serially.
这个旧 RTL 行为意味着一个 GMEM response word 会在 receiver LMEM writes 串行 replay 时占住 DXA drain path。

The old SimX multicast replay mirrored the same serial receiver-write model, so the original flat-speedup problem was an architectural performance issue rather than a SimX-only artifact.
旧 SimX multicast replay 也反映了同样的 serial receiver-write 模型，所以原始 flat-speedup 问题是架构性能问题，而不是单纯 SimX artifact。

The user's address-stride assumption is confirmed for the current intra-core cluster model.
用户关于地址 stride 的假设在当前 intra-core cluster 模型中成立。

SimX `sim/simx/cta_dispatcher.cpp` rounds each CTA LMEM allocation to `VX_CFG_MEM_BLOCK_SIZE`, pre-wraps the first CTA of a cluster so the whole cluster fits contiguously, and assigns members at contiguous aligned offsets.
SimX 的 `sim/simx/cta_dispatcher.cpp` 会把每个 CTA 的 LMEM allocation round 到 `VX_CFG_MEM_BLOCK_SIZE`，并在 cluster 首个 CTA 处预先 wrap，使整个 cluster 连续放置，成员位于连续 aligned offsets。

RTL `hw/rtl/core/VX_cta_dispatch.sv` implements the same cluster-contiguous LMEM reservation, so receiver addresses can be computed as `issuer_addr + rank * smem_stride` for same-core multicast.
RTL 的 `hw/rtl/core/VX_cta_dispatch.sv` 实现了相同的 cluster-contiguous LMEM reservation，所以 same-core multicast 的 receiver 地址可以用 `issuer_addr + rank * smem_stride` 计算。

The current host benchmark programs the multicast stride with `vortex::dxa::set_multicast(device, kDescSrc, local_mem)` and relies on `cluster_dim[0] = num_recv`.
当前 host benchmark 通过 `vortex::dxa::set_multicast(device, kDescSrc, local_mem)` 设置 multicast stride，并依赖 `cluster_dim[0] = num_recv`。

RTL completion detection currently sits in `hw/rtl/core/VX_mem_unit.sv`, where `lmem_dma_if.req_data.attr` is passed to `VX_dxa_completion`.
RTL completion detection 当前位于 `hw/rtl/core/VX_mem_unit.sv`，其中 `lmem_dma_if.req_data.attr` 被传给 `VX_dxa_completion`。

SimX completion detection currently sits in `sim/simx/cluster.cpp`, where the DXA-to-LMEM channel callback releases the barrier when a write packet carries `dxa_notify_done`.
SimX completion detection 当前位于 `sim/simx/cluster.cpp`，其中 DXA-to-LMEM channel callback 在 write packet 携带 `dxa_notify_done` 时 release barrier。

## Goal
## 目标

Close the multicast no-speedup problem with evidence, fixes, and new regression coverage.
用证据、修复和新的 regression coverage 闭环 multicast 没有 speedup 的问题。

Add a shared-memory atomic soft barrier path that can be compared against the existing low-overhead hard barrier.
添加 shared-memory atomic soft barrier 路径，用来和现有低开销 hard barrier 对比。

Produce experiment data that can support a poster claim about lower barrier overhead and lower stall time.
产出能支撑 poster 中关于 barrier overhead 更低、stall time 更短的实验数据。

## File Map
## 文件地图

`tests/regression/dxa_copy_mcast` remains the Figure 3(c) benchmark and multicast correctness vehicle.
`tests/regression/dxa_copy_mcast` 继续作为 Figure 3(c) benchmark 和 multicast correctness 载体。

`tools/dxa/run_copy_sweep.py` remains the sweep runner for Figure 3(c), including all 9 hardware configurations, 16 tile cases, and per-CTA versus multicast variants.
`tools/dxa/run_copy_sweep.py` 继续作为 Figure 3(c) sweep runner，覆盖 9 种硬件配置、16 种 tile case，以及 per-CTA versus multicast variants。

`tools/dxa/plot_copy_sweep.py` remains the Figure 3(c) plotting entry point and should be extended only if new counters or labels need to appear.
`tools/dxa/plot_copy_sweep.py` 继续作为 Figure 3(c) plotting 入口；只有在新 counters 或 labels 需要显示时才扩展。

`sim/simx/dxa/dxa_core.cpp` owns the SimX DXA worker model and must stop doing worker-local multicast receiver replay after the LMEM-side replay queue exists.
`sim/simx/dxa/dxa_core.cpp` 拥有 SimX DXA worker model；LMEM-side replay queue 存在后，它必须停止做 worker-local multicast receiver replay。

`sim/simx/cluster.cpp` owns the SimX DXA-to-LMEM channel binding and current completion callback, so it is the natural SimX insertion point for near-LMEM replay and completion timing.
`sim/simx/cluster.cpp` 拥有 SimX DXA-to-LMEM channel binding 和当前 completion callback，因此它是 SimX near-LMEM replay 和 completion timing 的自然插入点。

`sim/simx/types.h` owns `MemFlags`, so it must carry any SimX multicast replay metadata added to DXA LMEM write packets.
`sim/simx/types.h` 拥有 `MemFlags`，因此任何加到 DXA LMEM write packet 上的 SimX multicast replay metadata 都必须在这里承载。

`hw/rtl/dxa/VX_dxa_smem_wr.sv` currently owns RTL worker-local multicast replay and should be simplified to emit one multicast packet once the new replay module is active.
`hw/rtl/dxa/VX_dxa_smem_wr.sv` 当前拥有 RTL worker-local multicast replay；新 replay module 启用后，它应被简化为发出一个 multicast packet。

`hw/rtl/dxa/VX_dxa_pkg.sv` and `hw/rtl/VX_gpu_pkg.sv` own the RTL DXA/LMEM metadata widths and structs, so they must define the multicast replay sideband cleanly.
`hw/rtl/dxa/VX_dxa_pkg.sv` 和 `hw/rtl/VX_gpu_pkg.sv` 拥有 RTL DXA/LMEM metadata widths 与 structs，因此必须在这里干净定义 multicast replay sideband。

`hw/rtl/core/VX_mem_unit.sv` owns the LMEM DMA merge point and current completion detector instantiation, so it is the natural RTL insertion point for the near-LMEM replay engine.
`hw/rtl/core/VX_mem_unit.sv` 拥有 LMEM DMA merge point 和当前 completion detector instantiation，因此它是 RTL near-LMEM replay engine 的自然插入点。

`hw/rtl/dxa/VX_dxa_completion.sv` owns hard-barrier completion detection and should generate receiver completion events after the replay engine emits each receiver's final write.
`hw/rtl/dxa/VX_dxa_completion.sv` 拥有 hard-barrier completion detection，应在 replay engine 发出每个 receiver 的 final write 后生成 receiver completion event。

`sw/kernel/include/vx_barrier.h` owns the hard and soft barrier kernel-facing wrappers.
`sw/kernel/include/vx_barrier.h` 拥有 hard 和 soft barrier 的 kernel-facing wrappers。

`sw/kernel/include/vx_dxa.h` owns the DXA issue helpers and must expose hard-barrier and soft-barrier issue paths without changing ordinary hard-barrier kernel code.
`sw/kernel/include/vx_dxa.h` 拥有 DXA issue helpers，必须暴露 hard-barrier 与 soft-barrier issue paths，同时不改变普通 hard-barrier kernel code。

`tests/regression/smem_atomic_barrier` will validate LMEM AMO and pure soft-barrier semantics.
`tests/regression/smem_atomic_barrier` 将验证 LMEM AMO 和纯 soft-barrier semantics。

`tests/regression/dxa_copy -B` validates hard-barrier DXA completion timing, and `tests/regression/dxa_copy -B -S` validates the soft shared-memory atomic DXA completion path.
`tests/regression/dxa_copy -B` 验证 hard-barrier DXA completion timing，`tests/regression/dxa_copy -B -S` 验证 soft shared-memory atomic DXA completion path。

`tools/dxa/run_barrier_overhead.py` runs the C1 software-completion soft-barrier sweep and emits CSV.
`tools/dxa/run_barrier_overhead.py` 运行 C1 software-completion soft-barrier sweep，并输出 CSV。

`tools/dxa/plot_barrier_overhead.py` will render the absolute-overhead and overhead-ratio charts.
`tools/dxa/plot_barrier_overhead.py` 将渲染 absolute-overhead 和 overhead-ratio charts。

`tools/dxa/run_dxa_barrier_latency.py` runs the C2 hard-versus-soft DXA completion release-latency sweep with L2 enabled for every case.
`tools/dxa/run_dxa_barrier_latency.py` 运行 C2 hard-versus-soft DXA completion release-latency sweep，并且每个 case 都启用 L2。

`tools/dxa/plot_dxa_barrier_latency.py` renders the C2 hard-versus-soft DXA completion release-latency chart.
`tools/dxa/plot_dxa_barrier_latency.py` 渲染 C2 hard-versus-soft DXA completion release-latency chart。

## Phase A: Multicast Speedup Closure
## Phase A：Multicast Speedup 闭环

- [x] Treat SMEM-side multicast replay as the primary fix path, not merely an optional prototype.
- [x] 将 SMEM-side multicast replay 作为主要修复路径，而不是可选 prototype。

- [ ] Keep the multicast scope intra-core only; do not introduce cross-core multicast or DSM assumptions.
- [ ] multicast 范围保持 intra-core，不引入 cross-core multicast 或 DSM 假设。

- [x] Define a DXA-to-LMEM multicast packet format for the RTL DMA path: `data`, `byteen`, `base_addr`, `smem_stride`, `cta_mask` or `replay_count`, `bar_addr_base`, `last`, and `is_multicast`.
- [x] 为 RTL DMA path 定义 DXA-to-LMEM multicast packet format：`data`、`byteen`、`base_addr`、`smem_stride`、`cta_mask` 或 `replay_count`、`bar_addr_base`、`last` 和 `is_multicast`。

- [x] Extend `DXA_LMEM_ATTR_W` or add a sideband struct so the LMEM-side replay engine receives multicast metadata without stealing data payload bits.
- [x] 扩展 `DXA_LMEM_ATTR_W` 或增加 sideband struct，使 LMEM-side replay engine 能收到 multicast metadata，而不是占用 data payload bits。

- [x] Move receiver replay state from `hw/rtl/dxa/VX_dxa_smem_wr.sv` into a new near-LMEM module, tentatively `hw/rtl/dxa/VX_dxa_lmem_mcast_replay.sv`.
- [x] 将 receiver replay state 从 `hw/rtl/dxa/VX_dxa_smem_wr.sv` 移到新的 near-LMEM module，暂定为 `hw/rtl/dxa/VX_dxa_lmem_mcast_replay.sv`。

- [x] Place the new RTL replay module between the DXA DMA input and `lmem_dma_arb` or immediately after `lmem_dma_arb` before local memory bank write detection, whichever keeps TCU/DXA arbitration semantics clean.
- [x] 将新的 RTL replay module 放在 DXA DMA input 与 `lmem_dma_arb` 之间，或放在 `lmem_dma_arb` 后、local memory bank write detection 前；选择保持 TCU/DXA arbitration semantics 最干净的位置。

- [x] Preserve the single-DXA-read advantage: DXA worker should release its GMEM response slot after handing one multicast packet to the replay engine, not after all receiver writes have drained.
- [x] 保留 single-DXA-read 优势：DXA worker 在把一个 multicast packet 交给 replay engine 后就应释放 GMEM response slot，而不是等所有 receiver writes drain 完。

- [x] Let the near-LMEM replay engine generate per-receiver LMEM writes using `base_addr + rank * smem_stride` and the original byte enable.
- [x] 让 near-LMEM replay engine 使用 `base_addr + rank * smem_stride` 和原始 byte enable 生成每个 receiver 的 LMEM writes。

- [x] Generate one completion event for each receiver CTA only when that receiver's last replay write is accepted by LMEM.
- [x] 只有当某个 receiver CTA 的最后一个 replay write 被 LMEM 接收时，才为该 receiver CTA 生成一个 completion event。

- [x] Preserve hard-barrier early-credit semantics when the receiver completion fires before that CTA reaches `expect_tx`.
- [x] 当 receiver completion 早于该 CTA 到达 `expect_tx` 时，保持 hard-barrier early-credit semantics。

  Code support exists through signed hard-barrier event counters in SimX and RTL; Phase B still requires a forced early-completion stress to verify the intended schedule corner.
  代码层面已通过 SimX 和 RTL 的 signed hard-barrier event counter 支持该语义；Phase B 仍需要 forced early-completion stress 来验证目标调度角落。

- [x] Mirror the same architecture in SimX: replace `DxaCore::tick_worker_smem_wr` multicast per-CTA replay with a simulated LMEM-side replay queue, so SimX remains the oracle for RTL.
- [x] 在 SimX 中镜像同样架构：把 `DxaCore::tick_worker_smem_wr` 中的 multicast per-CTA replay 替换为 simulated LMEM-side replay queue，使 SimX 继续作为 RTL oracle。

- [ ] Add replay-engine counters: packets accepted from DXA, receiver writes emitted, replay cycles, replay queue occupancy, DXA slot-release cycles, and LMEM backpressure cycles.
- [ ] 增加 replay-engine counters：packets accepted from DXA、receiver writes emitted、replay cycles、replay queue occupancy、DXA slot-release cycles 和 LMEM backpressure cycles。

- [ ] Add SimX DXA counters for queue wait cycles, GMEM request issue cycles, GMEM response wait cycles, LMEM replay cycles, LMEM request-ready stall cycles, barrier release cycles, and per-transfer total latency.
- [ ] 增加 SimX DXA counters，记录 queue wait cycles、GMEM request issue cycles、GMEM response wait cycles、LMEM replay cycles、LMEM request-ready stall cycles、barrier release cycles 和 per-transfer total latency。

- [ ] Add RTL trace or perf-counter exposure for the same buckets around `VX_dxa_gmem_req`, `VX_dxa_smem_wr`, and `VX_dxa_completion`.
- [ ] 在 RTL 的 `VX_dxa_gmem_req`、`VX_dxa_smem_wr` 和 `VX_dxa_completion` 附近暴露同类 trace 或 perf counters。

- [ ] Measure per-CTA and multicast with `--writeback=none`, `--writeback=sample`, and a small `--writeback=full` correctness case while keeping L2 enabled.
- [ ] 在保持 L2 enabled 的前提下，用 `--writeback=none`、`--writeback=sample` 和一个小规模 `--writeback=full` correctness case 测量 per-CTA 与 multicast。

- [x] Verify whether multicast loses cycles primarily in receiver LMEM replay rather than GMEM reads.
- [x] 验证 multicast 是否主要在 receiver LMEM replay 而不是 GMEM reads 上损失 cycles。

  Current SimX evidence says very-small-tile multicast does **not** lose primarily in external receiver LMEM replay, because near-SMEM replay reduces DXA LMEM packets 4x but 16x16 cycles remain flat.
  当前 SimX 证据表明 very-small-tile multicast **不是** 主要输在外部 receiver LMEM replay，因为 near-SMEM replay 让 DXA LMEM packets 降低 4x，但 16x16 cycles 基本不变。

- [ ] Verify whether per-CTA's apparent advantage is caused by CTA scheduling, writeback policy, verification writeback, or forced serialization unrelated to the intended copy path.
- [ ] 验证 per-CTA 的表面优势是否来自 CTA scheduling、writeback policy、verification writeback，或与目标 copy path 无关的强制串行化。

- [x] Evaluate a double-buffered or prefetch-style multicast kernel mode that registers and issues the next multicast transfer before consuming or writing back the previous tile.
- [x] 评估一个 double-buffered 或 prefetch-style multicast kernel mode，在消费或写回前一个 tile 之前先 register 并 issue 下一个 multicast transfer。

  Result: correctness passes, but the current generic software pipeline path is slower than depth 1, so it is not the selected Figure 3(c) path.
  结果：正确性通过，但当前 generic software pipeline path 比 depth 1 更慢，因此不作为选定的 Figure 3(c) 路径。

- [ ] Evaluate transfer aggregation for small tiles so one multicast issue contains enough GMEM lines to keep the DXA/GMEM pipeline occupied.
- [ ] 评估 small tiles 的 transfer aggregation，让一次 multicast issue 包含足够多 GMEM lines，以保持 DXA/GMEM pipeline occupied。

- [ ] Disassemble per-CTA and multicast kernels and map the extra multicast instructions back to C++ source statements.
- [ ] 反汇编 per-CTA 与 multicast kernels，并把 multicast 多出来的指令对应回 C++ source statements。

- [ ] Remove or reduce avoidable software overhead in multicast helper construction if disassembly shows repeated non-copy work inside the tile loop.
- [ ] 如果反汇编显示 tile loop 内存在重复的非 copy 工作，则移除或降低 multicast helper construction 中可避免的软件开销。

- [x] Remove the current worker-local multicast replay from the performance path after the LMEM-side replay engine passes correctness.
- [x] 在 LMEM-side replay engine 通过正确性后，从 performance path 中移除当前 worker-local multicast replay。

- [x] Preserve one completion notification per receiver CTA after that receiver's final replay beat.
- [x] 保持每个 receiver CTA 在自己的 final replay beat 之后获得一次 completion notification。

- [ ] Compare the original serial replay datapath and the decoupled replay datapath on the same 4-CTA Figure 3(c) sweep.
- [ ] 在相同 4-CTA Figure 3(c) sweep 上比较原始 serial replay datapath 和 decoupled replay datapath。

- [ ] Produce the final Figure 3(c) CSV and plot only after the SMEM-side replay path is active in both SimX and RTLsim.
- [ ] 只有在 SimX 和 RTLsim 都启用 SMEM-side replay path 之后，才生成最终 Figure 3(c) CSV 和 plot。

- [ ] If RTLsim runtime prevents the full 288-row Figure 3(c) matrix, produce a complete SimX Figure 3(c) and an RTLsim representative matrix covering all `num_warps`/`num_threads` settings and all six barrier-overhead workload sizes.
- [ ] 如果 RTLsim runtime 阻碍完整 288-row Figure 3(c) matrix，就产出完整 SimX Figure 3(c)，以及覆盖所有 `num_warps`/`num_threads` 设置和六个 barrier-overhead workload size 的 RTLsim representative matrix。

## Phase B: No-Group-Sync Correctness
## Phase B：去掉 Group Sync 后的正确性

- [ ] Add a `dxa_copy_mcast` delay or schedule-stress mode that delays non-issuer receiver CTAs before `expect_tx`.
- [ ] 给 `dxa_copy_mcast` 增加 delay 或 schedule-stress mode，让非 issuer receiver CTA 在 `expect_tx` 之前故意延迟。

- [ ] Add a complementary stress mode that delays the issuer CTA after receivers expect the transaction.
- [ ] 增加互补 stress mode，让 issuer CTA 在 receivers expect transaction 之后延迟。

- [ ] Run both stress modes with `--mode=mcast --writeback=sample --num-ctas=4 --verify=1` at 512-by-512 and representative tile sizes.
- [ ] 用 512-by-512 和代表性 tile sizes 跑两个 stress mode，参数包含 `--mode=mcast --writeback=sample --num-ctas=4 --verify=1`。

- [x] Run a smaller `--writeback=full --verify=1` multicast case to verify full receiver data, not only sampled output.
- [x] 跑一个较小的 `--writeback=full --verify=1` multicast case，验证完整 receiver data，而不只验证 sample output。

- [ ] Confirm that signed event-credit semantics in SimX and RTL allow completion-before-expect without confusing valid large pending counts.
- [ ] 确认 SimX 和 RTL 中的 signed event-credit semantics 支持 completion-before-expect，同时不会和合法的大 pending count 混淆。

- [ ] Keep the old `group_barrier` behavior available only as a debug switch if it helps isolate regressions.
- [ ] 仅在有助于定位 regression 时，以 debug switch 保留旧 `group_barrier` 行为。

## Phase C: Shared-Memory Atomic Soft Barrier
## Phase C：Shared-Memory Atomic Soft Barrier

### Phase C0: LMEM AMO Enabling
### Phase C0：LMEM AMO 前置能力

- [x] Add explicit LMEM AMO support in SimX by allowing shared-memory AMO requests through `LocalMemSwitch` and implementing atomic RMW semantics in `LocalMem`.
- [x] 在 SimX 中添加明确的 LMEM AMO 支持：允许 shared-memory AMO requests 通过 `LocalMemSwitch`，并在 `LocalMem` 中实现 atomic RMW semantics。

- [x] Add explicit LMEM AMO support in RTL by preserving AMO sideband through `VX_lmem_switch.sv` and adding atomic RMW behavior inside or alongside `VX_local_mem.sv`.
- [x] 在 RTL 中添加明确的 LMEM AMO 支持：让 `VX_lmem_switch.sv` 保留 AMO sideband，并在 `VX_local_mem.sv` 内部或旁路添加 atomic RMW 行为。

- [x] Start with 32-bit AMOADD, AMOSWAP, AMOOR, AMOAND, AMOXOR, AMOMAX, AMOMINU, and AQRL variants needed by the counter or mask protocol.
- [x] 先支持 counter 或 mask protocol 需要的 32-bit AMOADD、AMOSWAP、AMOOR、AMOAND、AMOXOR、AMOMAX、AMOMINU 和 AQRL variants。

- [ ] Keep LR/SC local-memory correctness as a separate follow-up unless the chosen soft-barrier protocol requires compare-and-swap semantics.
- [ ] 除非选定的 soft-barrier protocol 需要 compare-and-swap 语义，否则将 LR/SC local-memory correctness 作为独立 follow-up。

### Phase C1: Software Soft-Barrier Benchmark
### Phase C1：纯软件 Soft-Barrier Benchmark

- [x] Add a `vortex::soft_barrier` or `vortex::smem_barrier` wrapper in `sw/kernel/include/vx_barrier.h`.
- [x] 在 `sw/kernel/include/vx_barrier.h` 中添加 `vortex::soft_barrier` 或 `vortex::smem_barrier` wrapper。

- [x] Keep the kernel-facing API parallel to `vortex::barrier`: construct or claim, call `expect_tx`, call `arrive`, `wait`, or `arrive_and_wait`.
- [x] 让 kernel-facing API 与 `vortex::barrier` 平行：construct 或 claim，调用 `expect_tx`，再调用 `arrive`、`wait` 或 `arrive_and_wait`。

- [x] Use a shared-memory state layout with an expected warp mask, arrived warp mask, signed event count, phase, and optional lock word.
- [x] 使用包含 expected warp mask、arrived warp mask、signed event count、phase 和可选 lock word 的 shared-memory state layout。

  The implemented C1 layout uses `events`, monotonic `arrived`, `phase`, and `expected_warps`; it deliberately avoids LR/SC-dependent mask reset.
  当前 C1 实现使用 `events`、monotonic `arrived`、`phase` 和 `expected_warps`；它有意避免依赖 LR/SC 的 mask reset。

- [x] Use signed `events` so DXA completion-before-expect has the same credit semantics as the hard barrier path.
- [x] 使用 signed `events`，让 DXA completion-before-expect 具有与 hard barrier path 一致的 credit semantics。

- [x] Implement `expect_tx(N)` as an atomic add of `N` to the soft barrier event count.
- [x] 将 `expect_tx(N)` 实现为对 soft barrier event count 的 atomic add `N`。

- [x] Implement DXA soft completion as an atomic add of `-1` to the soft barrier event count.
- [x] 将 DXA soft completion 实现为对 soft barrier event count 的 atomic add `-1`。

  This is implemented for the C1 software-completion benchmark; C2 still needs DXA hardware completion to emit the same decrement.
  这已经在 C1 software-completion benchmark 中实现；C2 仍需要 DXA hardware completion 发出同样的 decrement。

- [x] Implement `arrive` as an atomic set-bit operation on the arrived warp mask or an atomic increment if mask semantics are impossible.
- [x] 将 `arrive` 实现为对 arrived warp mask 的 atomic set-bit 操作；如果 mask semantics 不可行，再使用 atomic increment。

- [x] Implement `wait` as a loop that reads or atomically queries `events`, `arrived`, and `phase` until `events == 0` and all expected warps have arrived.
- [x] 将 `wait` 实现为循环读取或 atomic query `events`、`arrived` 和 `phase`，直到 `events == 0` 且所有 expected warps 到齐。

- [x] Implement one leader reset or phase-advance protocol so repeated barrier use cannot let late waiters observe a partially reset state.
- [x] 实现 one leader reset 或 phase-advance protocol，避免重复使用 barrier 时 late waiter 看到部分 reset 状态。

- [x] Avoid ordinary divergent `if (vx_thread_id() == 0)` control flow for warp-leader atomics; use `vx_tmc_one()` or an all-lane same-value pattern to avoid the divergence issue observed in the local AMO harness.
- [x] 避免用普通 divergent `if (vx_thread_id() == 0)` 控制流做 warp-leader atomics；使用 `vx_tmc_one()` 或 all-lane same-value pattern，避免 local AMO harness 中观察到的 divergence 问题。

- [x] Add a pure software workload mode that performs `expect_tx`, an explicit software completion decrement after a controlled LMEM copy/workload, and `arrive_and_wait`.
- [x] 添加一个纯软件 workload mode：执行 `expect_tx`，在受控 LMEM copy/workload 后显式执行 software completion decrement，然后 `arrive_and_wait`。

- [x] Use this C1 benchmark to quickly produce the first soft-barrier overhead trend, clearly labelled as software-completion soft barrier rather than DXA-completion soft barrier.
- [x] 使用这个 C1 benchmark 快速产出第一版 soft-barrier overhead trend，并明确标注为 software-completion soft barrier，而不是 DXA-completion soft barrier。

### Phase C2: DXA Soft-Completion Path
### Phase C2：DXA Soft-Completion 路径

- [x] Add a new DXA soft-barrier issue encoding or sideband because the current `vx_dxa_pack_meta` format cannot carry a pointer.
- [x] 增加新的 DXA soft-barrier issue encoding 或 sideband，因为当前 `vx_dxa_pack_meta` 格式无法携带 pointer。

- [x] Prefer a compact soft-barrier tag in the existing 27-bit raw barrier payload if it can encode the LMEM-local offset safely; otherwise add an explicit sideband in the DXA issue encoding.
- [x] 如果现有 27-bit raw barrier payload 能安全编码 LMEM-local offset，就优先使用 compact soft-barrier tag；否则在 DXA issue encoding 中增加显式 sideband。

- [x] Reserve a soft-barrier kind bit that cannot collide with ordinary hard-barrier ids, and document the exact bit layout in `sw/kernel/include/vx_dxa.h`, SimX decode, and RTL decode.
- [x] 保留一个不会与普通 hard-barrier id 冲突的 soft-barrier kind bit，并在 `sw/kernel/include/vx_dxa.h`、SimX decode 和 RTL decode 中记录精确 bit layout。

- [x] Update SimX DXA request metadata to carry `bar_kind` and either a hard barrier id or a soft barrier LMEM address.
- [x] 更新 SimX DXA request metadata，使其携带 `bar_kind`，以及 hard barrier id 或 soft barrier LMEM address 二者之一。

- [x] Update RTL DXA request structs in `VX_dxa_pkg.sv` and decode in `VX_dxa_unit.sv` to carry the same `bar_kind` and soft-barrier address.
- [x] 更新 RTL 的 `VX_dxa_pkg.sv` request structs 和 `VX_dxa_unit.sv` decode，使其携带同样的 `bar_kind` 与 soft-barrier address。

- [x] Update DXA completion in SimX and RTL so hard completions release `VX_bar_unit`, while soft completions perform LMEM atomic decrement.
- [x] 更新 SimX 和 RTL 的 DXA completion，让 hard completions release `VX_bar_unit`，soft completions 执行 LMEM atomic decrement。

- [ ] In SimX, route soft completions through the local-memory timing path when practical, not only a zero-cycle host-side counter mutation.
- [ ] 在 SimX 中，只要可行，就让 soft completion 走 local-memory timing path，而不是零周期 host-side counter mutation。

- [x] In RTL, decide whether the soft completion decrement is emitted by `VX_dxa_completion`, piggybacked through `VX_dxa_smem_wr`, or handled by a small near-LMEM completion micro-op generator.
- [x] 在 RTL 中决定 soft completion decrement 由 `VX_dxa_completion` 发出、通过 `VX_dxa_smem_wr` piggyback，还是由一个小型 near-LMEM completion micro-op generator 处理。

- [x] Add `tests/regression/smem_atomic_barrier` for pure shared-memory atomic barrier correctness and reuse.
- [x] 添加 `tests/regression/smem_atomic_barrier`，用于纯 shared-memory atomic barrier correctness 与复用。

- [x] Add or parameterize an existing DXA copy test for hard-barrier versus soft-barrier DXA completion comparison.
- [x] 添加或参数化现有 DXA copy test，用来比较 hard-barrier 与 soft-barrier DXA completion。

- [x] Keep all new soft-barrier tests small enough to run first in SimX before RTL execution.
- [x] 让所有新的 soft-barrier tests 足够小，先能在 SimX 中运行，再进入 RTL。

## Phase D: Barrier Overhead Experiment Shape
## Phase D：Barrier Overhead 实验形态

- [ ] Define `T_register` as the cycle when `expect_tx` has become visible to the barrier state.
- [ ] 将 `T_register` 定义为 `expect_tx` 已经对 barrier state 可见的 cycle。

- [ ] For the hard barrier, record `T_register` when the WCTL/barrier event-attach update is accepted by `VX_bar_unit` or the SimX `BarrierUnit`.
- [ ] 对 hard barrier，在 WCTL/barrier event-attach update 被 `VX_bar_unit` 或 SimX `BarrierUnit` 接收时记录 `T_register`。

- [ ] For the soft barrier, record `T_register` when the shared-memory atomic add to the soft barrier event counter commits.
- [ ] 对 soft barrier，在 shared-memory atomic add 提交到 soft barrier event counter 时记录 `T_register`。

- [ ] Also record `T_issue` when the DXA issue instruction is accepted, but do not use it as the primary overhead denominator unless registration and issue are fused in that implementation.
- [ ] 同时记录 DXA issue instruction 被接收的 `T_issue`，但除非某个实现中 registration 和 issue 被融合，否则不把它作为主要 overhead denominator。

- [ ] Record `T_event_complete` at SMEM/LMEM completion detection for the last write belonging to the async event.
- [ ] 在 async event 的最后一个 write 到达 SMEM/LMEM completion detection 时记录 `T_event_complete`。

- [ ] Record `T_arrivals_complete` when all participating warps or the expected warp mask have arrived at the barrier.
- [ ] 在所有参与 warps 或 expected warp mask 到达 barrier 时记录 `T_arrivals_complete`。

- [ ] Record `T_release` when the waiting warp is actually released by the hard barrier or when the soft wait loop observes the release condition and exits.
- [ ] 在 waiting warp 被 hard barrier 实际释放，或 soft wait loop 观察到 release condition 并退出时记录 `T_release`。

- [ ] Define absolute barrier overhead as `T_release - max(T_event_complete, T_arrivals_complete)`.
- [ ] 将绝对 barrier overhead 定义为 `T_release - max(T_event_complete, T_arrivals_complete)`。

- [ ] Report the registration-normalized deviation as `(T_release - T_register) - (T_event_complete - T_register)`, and label it as equivalent to `T_release - T_event_complete` only when arrivals are already complete.
- [ ] 报告 registration-normalized deviation：`(T_release - T_register) - (T_event_complete - T_register)`；只有当 arrivals 已经完成时，才标注它等价于 `T_release - T_event_complete`。

- [ ] Verify the user's hypothesis that hard-barrier absolute overhead stays nearly constant across workload sizes.
- [ ] 验证用户假设：hard-barrier absolute overhead 在不同 workload size 下基本保持恒定。

- [ ] Verify that soft-barrier overhead is high for small workloads because polling atomics/loads compete with DXA writeback traffic on shared memory.
- [ ] 验证 soft-barrier 在小 workload 下 overhead 很高，是因为 polling atomics/loads 与 DXA writeback traffic 竞争 shared memory。

- [ ] Sweep workload byte sizes of 1KB, 2KB, 4KB, 8KB, 16KB, and 32KB per async event.
- [ ] sweep 每个 async event 的 workload byte sizes：1KB、2KB、4KB、8KB、16KB 和 32KB。

- [ ] For 32-bit element tests, use representative tile shapes `16x16`, `16x32`, `32x32`, `32x64`, `64x64`, and `64x128` to produce 1KB, 2KB, 4KB, 8KB, 16KB, and 32KB payloads.
- [ ] 对 32-bit element tests，使用代表性 tile shapes `16x16`、`16x32`、`32x32`、`32x64`、`64x64` 和 `64x128`，分别产生 1KB、2KB、4KB、8KB、16KB 和 32KB payload。

- [ ] Collect the same timing buckets for hard barrier and shared-memory atomic soft barrier.
- [ ] 对 hard barrier 和 shared-memory atomic soft barrier 收集相同 timing buckets。

- [ ] Record atomic instruction count, LMEM AMO requests, wait-loop iterations, and cycles spent spinning for the soft barrier.
- [ ] 记录 soft barrier 的 atomic instruction count、LMEM AMO requests、wait-loop iterations 和 spin cycles。

- [ ] Record hard barrier event attach/release cycles and stall cycles for the hard barrier.
- [ ] 记录 hard barrier 的 event attach/release cycles 和 stall cycles。

- [ ] Compute overhead ratio as `absolute_barrier_overhead / max(1, T_release - T_register)`.
- [ ] 将 overhead ratio 计算为 `absolute_barrier_overhead / max(1, T_release - T_register)`。

- [ ] Plot hard versus soft absolute overhead as a grouped bar chart over the six workload sizes.
- [ ] 画 hard versus soft absolute overhead grouped bar chart，横轴为六个 workload sizes。

- [ ] Plot hard versus soft overhead ratio as a line or grouped bar chart over the same six workload sizes.
- [ ] 在相同六个 workload sizes 上画 hard versus soft overhead ratio line chart 或 grouped bar chart。

- [ ] The expected poster trend is that hard-barrier absolute overhead is almost flat, while soft-barrier overhead ratio is largest at 1KB and decreases as payload grows.
- [ ] 期望 poster trend 是 hard-barrier absolute overhead 几乎平坦，而 soft-barrier overhead ratio 在 1KB 时最大，并随 payload 增大而下降。

- [ ] If the measured trend disagrees with the expectation, keep the data and add counters for LMEM AMO traffic, DXA LMEM writes, and wait-loop iterations before changing the claim.
- [ ] 如果 measured trend 与预期不一致，保留数据，并先增加 LMEM AMO traffic、DXA LMEM writes 和 wait-loop iterations counters，再调整 claim。

## Morning Delivery Targets
## 明早交付目标

- [ ] Produce a poster-quality Figure 3(c) multicast speedup plot from the fixed SMEM-side replay architecture.
- [ ] 从修复后的 SMEM-side replay 架构产出 poster-quality 的 Figure 3(c) multicast speedup plot。

- [ ] Produce the CSV used for the Figure 3(c) plot and record the exact SimX/RTLsim commands used to generate it.
- [ ] 产出 Figure 3(c) plot 使用的 CSV，并记录生成它的准确 SimX/RTLsim 命令。

- [ ] Ensure SimX and RTLsim both implement the same multicast replay architecture and pass the same correctness smoke before final plotting.
- [ ] 确保 SimX 和 RTLsim 都实现同一个 multicast replay 架构，并在最终画图前通过相同 correctness smoke。

- [ ] Produce a hard-versus-soft barrier absolute-overhead grouped bar chart across 1KB, 2KB, 4KB, 8KB, 16KB, and 32KB payloads.
- [ ] 产出 hard-versus-soft barrier absolute-overhead grouped bar chart，覆盖 1KB、2KB、4KB、8KB、16KB 和 32KB payload。

- [ ] Produce a hard-versus-soft barrier overhead-ratio trend chart across the same workload sizes.
- [ ] 产出 hard-versus-soft barrier overhead-ratio trend chart，覆盖相同 workload sizes。

- [ ] Include a short root-cause note explaining that the old multicast path allowed GMEM outstanding requests but held slot release behind worker-local receiver replay.
- [ ] 附上一份短 root-cause note，解释旧 multicast path 虽然允许 GMEM outstanding requests，但 slot release 被 worker-local receiver replay 阻塞。

- [ ] Include a short architecture note explaining that the fixed path offloads receiver replay to near-LMEM logic using cluster-contiguous LMEM placement.
- [ ] 附上一份短 architecture note，解释修复后的路径利用 cluster-contiguous LMEM placement，把 receiver replay offload 到 near-LMEM logic。

## Verification Commands
## 验证命令

Run commands from `build/` to respect the repo workflow.
所有命令都从 `build/` 目录运行，以遵守 repo workflow。

Use a timeout wrapper for long simulations, but increase timeout only after trace evidence shows progress rather than deadlock.
长仿真使用 timeout wrapper，但只有 trace 证据显示仍在前进而非 deadlock 时才增加 timeout。

```bash
env CONFIGS="-DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_DXA_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=16 --app=dxa_copy_mcast --args="--mode=mcast --writeback=sample --num-ctas=4 -r 512 -c 512 -R 16 -C 16 --verify=1"
```

```bash
env CONFIGS="-DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_DXA_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=16 --app=dxa_copy_mcast --args="--mode=mcast --writeback=full --num-ctas=4 -r 64 -c 64 -R 16 -C 16 --verify=1"
```

Run the complete SimX Figure 3(c) matrix after the SMEM-side multicast replay path is active:
在 SMEM-side multicast replay path 启用后，运行完整 SimX Figure 3(c) matrix：

```bash
python3 ../tools/dxa/run_copy_sweep.py --build-dir . --driver simx --figure 3c --matrix-size 512 --warps 8,16,32 --threads 8,16,32 --tile-sizes 16,32,64,128 --num-ctas 4 --mcast-writeback sample --verify 1 --timeout 120 --output ../docs/results/dxa_copy_sweep/simx_3c_smem_replay.csv
```

Render the SimX Figure 3(c) plot:
渲染 SimX Figure 3(c) plot：

```bash
python3 ../tools/dxa/plot_copy_sweep.py ../docs/results/dxa_copy_sweep/simx_3c_smem_replay.csv --figure 3c --out-dir ../docs/results/dxa_copy_sweep
```

Run the RTLsim Figure 3(c) matrix after the RTL SMEM-side replay path passes smoke:
在 RTL SMEM-side replay path 通过 smoke 后，运行 RTLsim Figure 3(c) matrix：

```bash
python3 ../tools/dxa/run_copy_sweep.py --build-dir . --driver rtlsim --figure 3c --matrix-size 512 --warps 8,16,32 --threads 8,16,32 --tile-sizes 16,32,64,128 --num-ctas 4 --mcast-writeback sample --verify 1 --timeout 120 --output ../docs/results/dxa_copy_sweep/rtlsim_3c_smem_replay.csv
```

Run local-memory AMO smoke tests before any soft-barrier performance claim:
在做任何 soft-barrier 性能 claim 之前，先运行 local-memory AMO smoke tests：

```bash
env CONFIGS="-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_DXA_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=simx --cores=1 --warps=1 --threads=4 --app=amo --args="-tamoadd -tamoadd_aqrl -tamoswap -n4 -l -c"
```

```bash
env CONFIGS="-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_DXA_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=1 --threads=4 --app=amo --args="-tamoadd -tamoadd_aqrl -tamoswap -n4 -l -c"
```

Run the local-memory AMO bitwise/minmax group to catch sideband or byte-offset regressions:
运行 local-memory AMO bitwise/minmax group，以捕获 sideband 或 byte-offset regressions：

```bash
env CONFIGS="-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_DXA_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=simx --cores=1 --warps=1 --threads=4 --app=amo --args="-tamoor -tamoand -tamoxor -tamomax -tamominu -n4 -l -c"
```

```bash
env CONFIGS="-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_DXA_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=1 --threads=4 --app=amo --args="-tamoor -tamoand -tamoxor -tamomax -tamominu -n4 -l -c"
```

```bash
env CONFIGS="-DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_A_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=16 --app=smem_atomic_barrier --args="--mode=arrive-wait --verify=1"
```

```bash
env CONFIGS="-DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_A_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=16 --app=dxa_copy --args="-d2 -s0 16 -s1 16 -t0 16 -t1 16 -B"
```

```bash
env CONFIGS="-DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_A_ENABLE" timeout -k 5s 120s ./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=8 --l2cache --perf=16 --app=dxa_copy --args="-d2 -s0 16 -s1 16 -t0 16 -t1 16 -B -S"
```

Run the DXA hard-versus-soft barrier workload sweep with L2 enabled:
在启用 L2 的情况下运行 DXA hard-versus-soft barrier workload sweep：

```bash
python3 ../tools/dxa/run_dxa_barrier_latency.py --build-dir . --driver simx --workloads 1024,2048,4096,8192,16384,32768 --warps 8 --threads 8 --timeout 120 --output ../docs/results/dxa_barrier_overhead/simx_dxa_barrier_latency.csv
```

```bash
python3 ../tools/dxa/run_dxa_barrier_latency.py --build-dir . --driver rtlsim --workloads 1024,2048,4096,8192,16384,32768 --warps 8 --threads 8 --timeout 120 --output ../docs/results/dxa_barrier_overhead/rtlsim_dxa_barrier_latency.csv
```

Render the DXA hard-versus-soft barrier plots:
渲染 DXA hard-versus-soft barrier plots：

```bash
python3 ../tools/dxa/plot_dxa_barrier_latency.py ../docs/results/dxa_barrier_overhead/simx_dxa_barrier_latency.csv --out-dir ../docs/results/dxa_barrier_overhead
```

```bash
python3 ../tools/dxa/plot_dxa_barrier_latency.py ../docs/results/dxa_barrier_overhead/rtlsim_dxa_barrier_latency.csv --out-dir ../docs/results/dxa_barrier_overhead
```

## Deliverables
## 交付物

- [ ] Updated multicast performance counters and CSV summaries under `docs/results/dxa_copy_sweep/`.
- [ ] 更新后的 multicast performance counters 和 CSV summaries，放在 `docs/results/dxa_copy_sweep/` 下。

- [ ] A root-cause note explaining why multicast was slower or flat before the fix.
- [ ] 一份 root-cause note，解释修复前 multicast 为什么更慢或没有 speedup。

- [ ] A corrected multicast datapath or a documented architectural limitation with poster-safe wording.
- [ ] 一个修正后的 multicast datapath，或一个有 poster-safe wording 的架构限制说明。

- [ ] No-group-sync correctness stress results for SimX and RTL when RTL runtime is practical.
- [ ] SimX 和可行 RTL runtime 下的 no-group-sync correctness stress 结果。

- [x] Shared-memory atomic support in SimX and RTL sufficient for the soft barrier.
- [x] SimX 和 RTL 中足够支持 soft barrier 的 shared-memory atomic support。

- [ ] A hard-barrier versus soft-barrier comparison table with transaction duration, release duration, and derived barrier overhead.
- [ ] 一张 hard-barrier 与 soft-barrier 对比表，包含 transaction duration、release duration 和推导出的 barrier overhead。

- [ ] Poster-ready plots or tables for multicast speedup and barrier overhead.
- [ ] 可用于 poster 的 multicast speedup 和 barrier overhead plots 或 tables。

## Stop Conditions
## 停止条件

Stop a full sweep when many multicast cases show no speedup before instrumentation explains the lost cycles.
如果大量 multicast cases 在 instrumentation 解释 lost cycles 之前没有 speedup，就停止 full sweep。

Do not continue producing flat-speedup data as if it were final experimental evidence.
不要继续把没有 speedup 的平坦数据当成最终实验依据产出。

Stop soft-barrier performance claims until shared-memory AMO correctness passes in both the software model and the RTL path being measured.
在 software model 和被测 RTL path 中 shared-memory AMO correctness 通过之前，停止 soft-barrier 性能 claim。

Do not compare hard barrier against a global-memory atomic barrier and describe it as shared-memory atomic unless the text explicitly labels it as a temporary baseline.
不要把 hard barrier 和 global-memory atomic barrier 的比较描述成 shared-memory atomic，除非文字明确标注这是 temporary baseline。

## Immediate Next Steps
## 立即下一步

- [x] Run the existing no-`group_sync` multicast sample correctness smoke with L2 enabled.
- [x] 运行现有 no-`group_sync` multicast sample correctness smoke，并启用 L2。

- [ ] Add forced early-completion multicast stress before trusting completion-before-expect as fully covered.
- [ ] 在完全信任 completion-before-expect 之前，添加 forced early-completion multicast stress。

- [x] Add SimX counters around multicast LMEM replay and GMEM wait to decide whether the first fix should target replay decoupling or software instruction overhead.
- [x] 在 multicast LMEM replay 和 GMEM wait 周围添加 SimX counters，以决定第一个修复应针对 replay decoupling 还是软件指令 overhead。

  The first SimX fix has already targeted replay decoupling and near-SMEM replay; the next instrumentation pass should separate short-transfer pipeline underfill from software instruction overhead.
  第一轮 SimX 修复已经针对 replay decoupling 和 near-SMEM replay；下一轮 instrumentation 应区分 short-transfer pipeline underfill 和 software instruction overhead。

- [ ] Inspect generated assembly for per-CTA and multicast kernels after building the current source.
- [ ] 在构建当前源码后，检查 per-CTA 和 multicast kernels 的 generated assembly。

- [x] Implement shared-memory AMO correctness in SimX before touching soft-barrier DXA completion.
- [x] 先在 SimX 中实现 shared-memory AMO correctness，再修改 soft-barrier DXA completion。

- [x] Implement shared-memory AMO correctness in RTL before using RTLsim for soft-barrier measurements.
- [x] 先在 RTL 中实现 shared-memory AMO correctness，再用 RTLsim 做 soft-barrier measurements。

- [ ] Implement the soft-barrier wrapper only after the LMEM AMO smoke test exists.
- [ ] 只有在 LMEM AMO smoke test 存在之后，才实现 soft-barrier wrapper。
