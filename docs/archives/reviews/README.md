# Vortex Graphics v2 — End-to-End Review (2026-06-17)

In-depth review of the graphics-v2 + PRISM-RTU implementation across the full stack,
focused on correctness, efficiency, performance, and alignment with mainstream
NVIDIA / AMD / Intel GPU design methodology (the "true GPU" goal).

**Start here:** [review_v2.1_recommendations.md](review_v2.1_recommendations.md) — the
cross-cutting synthesis, themes, and the prioritized (P0/P1/P2) v2.1 roadmap.

### Per-area reviews

| Area | Doc | Grade |
|---|---|:---:|
| 1. mesa_vortex SW stack (vortexpipe driver) | [review_mesa_vortex_sw.md](review_mesa_vortex_sw.md) | D |
| 2. prism_v3 graphics runtime stack | [review_gfx_runtime.md](review_gfx_runtime.md) | B− |
| 3. prism_v3 graphics kernel stack | [review_gfx_kernel.md](review_gfx_kernel.md) | B |
| 4. prism_v3 RTU runtime stack | [review_rtu_runtime.md](review_rtu_runtime.md) | B−/C |
| 5. prism_v3 RTU kernel stack | [review_rtu_kernel.md](review_rtu_kernel.md) | A− |
| 6. prism_v3 graphics SimX implementation | [review_gfx_simx.md](review_gfx_simx.md) | B/B+ |
| 7. prism_v3 RTU SimX implementation | [review_rtu_simx.md](review_rtu_simx.md) | B |
| 8. prism_v3 RTU RTL implementation | [review_rtu_rtl.md](review_rtu_rtl.md) | B/B+ |

### Headline outcomes
- **§8 multi-draw determinism bug root-caused** — a SimX SFU `vx_om4` window-handoff hazard
  (not scratch residual); fix is P0-1.
- **SimX↔RTL RTU timing model is ~10× optimistic** (models a 4-wide PE array the RTL lacks) —
  P1-1, the highest-leverage fix.
- **v2 ABI is half-migrated** — host+SimX on the new bin/`vx_om4`/`vx_tex4` ABI; RTL RASTER
  and the Mesa driver are not.
