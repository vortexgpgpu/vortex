# Latest S2G timeline traces

These three figures were regenerated on 2026-09-08 from the current debug build on `chengxuan@orcas2.cs.ucla.edu`, using the same `16x16x32`, 16-iteration, 3/3/3-stage kernel and L2-enabled configuration.

| Figure | Mode | Remote trace | Result |
|---|---|---|---|
| `latest_mode0_tma_s2g.png` | TMA load + S2G store | `latest-full-mode0.trace.log` | PASS |
| `latest_mode1_tma_global.png` | TMA load + staged LSU global store | `latest-full-mode1.trace.log` | PASS |
| `latest_mode2_lsu_global.png` | LSU load + global store | `latest-full-mode2.trace.log` | PASS |

The plots use 2,000-cycle bins and a 31-bin moving average. They show frontend issue share and accepted-request line-equivalent rates; they are not direct measurements of FU busy occupancy or physical DRAM payload bandwidth. The S2G and staged-global traces are therefore comparable as request/issue timelines, but the figure must not be described as a calibrated TCU-utilization result.

The corresponding performance runs report 70,309, 76,381, and 314,227 cycles for modes 0, 1, and 2 respectively. The full logs are retained at `/home/chengxuan/codex-runs/simple-groups-ws.P8caul/build-trace/tests/regression/dxa_tma_wgmma_s2g_pipeline/`.
