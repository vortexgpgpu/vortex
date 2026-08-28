# V80 Stabilization: Full Issue Inventory, Fix Proposal, and Execution Plan

Date: 2026-08-28. Status: proposal (investigation complete; execution gated on
the rstmin8 build already in flight).

## 0. Why progress stalled — an honest accounting

The last ~12 hours were spent chasing ONE symptom (the staged readback
returning stale data / the completion line stuck one behind Q_SEQNUM) through
seven transport-level theories, each requiring a ~38 min Vivado build to test
on silicon. Every theory was wrong because the bug was never in the transport:
`VX_cp_engine` handed the completion writer the *pre-increment* sequence
number, so the memory completion line was ≡ N−1 **by construction** and no
amount of write-path fixing could ever change what value was written.

The process failure: theories were tested serially on silicon at ~40 min per
iteration, when the invariant "the fence never observes cmpl == seqnum, ever,
under any transport change" should have forced a value-path audit far sooner.
The corrective in this plan: **no more silicon iterations to test theories.**
Every remaining hypothesis is either (a) already answered by evidence on disk,
(b) answerable in simulation, or (c) folded into the one build already
running. At most one further build (the full-size config) follows, and only
after the minimal config fully validates.

## 1. Issue inventory (everything currently open, with evidence)

### A. Completion line off-by-one — ROOT CAUSE FOUND, fix built, silicon-pending
- Symptom: every fence wait warns `completion line (N−1) never caught up to
  Q_SEQNUM (N)` — 35/35 waits in each of the last four demo runs (fp_1..4
  logs), perfectly deterministic.
- Cause: `retire_seqnum = seqnum_r` (pre-increment) in `VX_cp_engine.sv` while
  Q_SEQNUM reads the post-retire count.
- Fix: commit 3420c76ed — `retire_seqnum = seqnum_r + 64'd1`. Verified present
  in the rstmin8 build tree (`iprepo/vortex_afu/src/VX_cp_engine.sv:138`).
  Unit tests corrected (they had encoded the bug: cp_core expected cmpl==0
  after one retire). Sim green.
- Remaining: silicon confirmation on rstmin8. Acceptance: **zero** fence
  warnings across demo ×12.

### B. Launch retires before stores drain — fix built, silicon-pending
- Symptom: intermittent stale readback of kernel results (the original demo
  Heisenbug; closed by any tracing).
- Cause: Vortex deasserts `busy` when warps retire while stores are still
  draining through the caches; the CP retires the launch and the chained
  download DMA reads device memory early.
- Fix: same commit — `cp_gpu_if.busy = vx_busy || !mem_idle_all` in
  `VX_afu_wrap.sv:437`. In the rstmin8 tree.
- Remaining: silicon confirmation. Acceptance: demo ×12 = 12/12, then the
  sweep ladder.

### C. QDMA D2H staleness + the uncommitted H2D "kick" — status DISPUTED, must be A/B tested
- `sw/runtime/aved/vortex.cpp` (uncommitted, edited 03:54 today) adds an H2D
  write of the 64 KB scratch region before every staged refresh, with a
  comment claiming this is "the kick that actually works" because D2H reads
  returned pre-existing data until an H2D write transited the path.
- That comment **overclaims**: it was written before the off-by-one was found,
  and the observation it cites (cmpl advancing only after a doorbell's H2D) is
  fully explained by the off-by-one (cmpl legitimately advances to N−1 at the
  *next* retire, which a doorbell triggers). Whether QDMA D2H actually has a
  freshness hazard is **unknown** — the off-by-one contaminated every
  observation that suggested it.
- Plan: once rstmin8 shows fence=0/demo 12/12 **with** the kick, remove the
  kick and re-run demo ×12. If still clean, the kick was a dead theory like
  the others and is deleted. If failures return, the QDMA hazard is real, the
  kick stays, and the comment is rewritten with the A/B evidence.

### D. Dead-theory hardening accumulated in the CP/runtime — strip after validation
Added under theories now known or suspected dead, each individually harmless:
1. `VX_cp_completion` SHOVE states (second write of seqnum to cmpl+64).
2. `VX_cp_dma` S_FLUSH_AR/R read-back states on host-direction downloads.
3. Runtime 500 µs fence backoff + 50 ms cap (the cap currently *masks* fence
   failure — 4/4 demo runs "PASSED" only via ~35 timeouts ≈ 1.75 s of dead
   time per run).
4. 128 KB `staged_buster_` eviction region + its D2H read each refresh.
5. 2-line cmpl allocation (required by the shove; stays iff shove stays).
- Plan (all decidable WITHOUT extra builds, because removal is software or can
  ride the final full build): after C's A/B, strip 3's silent fallback — a
  fence timeout becomes a loud error, because with A fixed a timeout means
  something is genuinely broken. Items 1, 2, 5 are in the rstmin8/full RTL
  already; they are benign hardening — remove 2 (pure dead code) in the full
  build, keep 1+5 only if documented as belt-and-braces, else remove. Item 4:
  remove with C if C's kick is removed.

### E. Legacy AXI-Lite debug window reads garbage on silicon — remove, don't chase
- `VX_afu_ctrl` 0x40/0x44 read zeros and 0x4C a phantom 0x1 on silicon even
  after the implicit-net fix, while the identical counters read byte-exact
  through the CP window (0x1030/0x1034). The legacy window is superseded;
  root-causing it buys nothing. Plan: delete the legacy-window copies of the
  debug regs in the full build; keep the CP-window ones (they exonerated the
  AFU in one measurement and cost ~40 LUTs).

### F. Uncommitted DXA fine/coarse shifter split (VX_dxa_smem_wr.sv, dated Aug 1)
- Pre-existing gfxw timing work, silently present in EVERY build of this saga
  (sim and silicon) but never explicitly validated or committed. Orthogonal to
  V80 bring-up but a landmine: it is in the rstmin8 netlist and would be in
  the full build. Plan: run the DXA/TCU sim tests (sgemm_tcu on simx +
  xrtsim) against it; if green, commit it with its own message; if not
  immediately green, `git stash` it OUT of the tree before the full build so
  the release bitstream contains only committed RTL. It must not stay
  half-adopted.

### G. Other uncommitted changes — commit now
- `jtag_load_vortex.sh` realpath fix: correct, tested, commit as-is.
- `vortex.cpp` kick: hold for C's A/B, then commit either the kick with honest
  evidence or its removal.

### H. Full-size configuration is 8+ commits stale
- The rescued rst3 vbin predates the off-by-one fix, busy fix, req_gate,
  full-line completion, drain counters, and implicit-net fixes. Plan: ONE
  final full build, started only after the minimal config passes the whole
  ladder, containing the post-cleanup RTL (D/E decisions applied) so it is
  built exactly once.

### I. Reset architecture — ACCEPTED ON SILICON, one demo outstanding
- Twice-in-one-boot, refusal path, queue reset, AFU reload over a used AFU,
  and killed-process recovery are all demonstrated. Outstanding: recovery from
  a *livelocked kernel* (SIGKILL mid-kernel, not post-doorbell). Plan: after
  rstmin8 validates, run the wgmma-style livelock (or an infinite-loop kernel)
  and demonstrate reset recovery; add to reset_acceptance.sh.

### J. Known-benign build noise (documented so nobody chases it again)
- `ModularNoC 90-3` CRITICAL WARNING appears in every good build (rstmin6/7/8)
  — partial-reconfig flow artifact.
- No implicit-net warnings remain for our modules in the rstmin8 synth log
  (checked; that class of bug burned us four times).

### K. Backlog explicitly out of scope for stabilization
~/dev/v80 script cleanup (needs user go-ahead), SLASH fork push (needs
credentials), upstream packaging fixes, KERNEL_FREQ 150-vs-200, deleting
/opt/xilinx/slash, toolchain_install.sh --slash. None block the above; listed
so they are not silently dropped.

## 2. Execution plan

Ordered; each phase has a hard gate. No phase starts a Vivado build except
phase 5, and phase 5 starts exactly one.

**Phase 0 — now, while rstmin8 builds (no hardware, no builds)**
1. Commit the jtag realpath fix.
2. Run the DXA sim tests (issue F) on simx/xrtsim; commit or stash the DXA
   diff on the result.
3. Re-run cp_engine/cp_core unit tests + avedsim demo/sgemv/sgemm as a final
   pre-silicon regression of 3420c76ed.
   Gate: all green, tree contains only deliberate changes.

**Phase 1 — rstmin8 minimal validation (vbin lands ~04:45)**
4. `jtag_load_vortex.sh` rstmin8; fence-health run:
   `VORTEX_AVED_FENCE_DEBUG=1 run_hw_test.sh demo` → **expect 0 "never caught
   up"** (first time ever). This single number confirms issue A on silicon.
5. demo ×12 → expect 12/12 (confirms B; no timeout fallback masking, per the
   fence count).
6. `hw_sweep.sh minimal vecadd demo sgemv sgemm`, then `reset_acceptance.sh`.
   Gate: all pass. If the fence count is nonzero, STOP — the off-by-one
   diagnosis is incomplete; re-audit the value path in sim before any build.

**Phase 2 — A/B the H2D kick (software only, issue C)**
7. Remove the kick, demo ×12. Keep-or-delete on the result; commit vortex.cpp
   with the evidence in the message.

**Phase 3 — strip dead hardening (software + RTL edits, no build yet; D, E)**
8. Fence timeout → hard error; remove backoff if fence is instant; remove
   buster region if the kick died; remove DMA flush states + legacy debug
   window in RTL (rides the phase-5 build). Re-run sim regression.

**Phase 4 — livelocked-kernel recovery demo (I)** on rstmin8 silicon; extend
reset_acceptance.sh.

**Phase 5 — the one full-size build**
9. Start the full config with the post-cleanup tree. During the build: update
   docs (afu_reset_architecture_proposal, aved_driver_architecture, this file)
   and memory with the confirmed root cause; write the retro note on the
   seven dead theories.
10. Load, full ladder (fence health, demo ×12, sweep, reset acceptance,
    sgemm OPTS=-n1024). Gate: all pass → V80 support is DONE; declare it with
    the evidence table.

**Failure policy:** any silicon surprise gets ONE fence-debug + drain-counter
measurement, then goes back to simulation for reproduction. No
theory-per-bitstream iteration, ever again.
