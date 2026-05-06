# Master → feature_master Merge Proposal

**Date:** 2026-05-03
**Status:** Draft
**Author:** Blaise Tine
**Scope:** Merge all changes added to `vortexgpgpu/vortex` `master` in the
past two weeks (since 2026-04-19, merge-base `8b10348e`) into the current
`feature_master` branch (`tinebp-patch-2`, simx_v3 line).
**Related:** [simx_v3_proposal.md](simx_v3_proposal.md),
[dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md),
[wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md).

---

## 1. Constraints (load-bearing)

Any approach that breaks one of these is wrong.

1. **Authorship preserved.** Every external contributor commit must land
   with its original `author` and `author-date` intact. The committer
   line will become `tinebp` because the replay happens locally; the
   `author` field must not.
2. **History preserved.** Each upstream PR must be reconstructible on
   the new branch as a recognizable unit (one PR ↔ one merge commit, or
   one PR ↔ one cherry-picked sequence with the original SHAs annotated
   via `cherry-pick -x`). A future reader must be able to ask "where did
   PR #320 land on simx_v3?" and find a single answer.
3. **No regressions on the simx_v3 line.** The merge must not undo any
   of the architectural decisions in
   [simx_v3_proposal.md](simx_v3_proposal.md),
   [dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md), or
   [wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md). In
   particular: no reintroduction of `core->mem_read`/`core->mem_write`
   bypasses, no monolithic `Emulator::execute()` switch, no shadow LMEM
   image.
4. **One-way only.** This merge brings master into feature_master. No
   reverse push to master is in scope; the simx_v3 reorg is not yet
   ready for upstream.

---

## 2. Why a plain `git merge` will not work

The two branches have diverged structurally since the merge-base
(`8b10348e`, 2025-07-11). The reorg deltas that conflict with almost
every incoming PR are:

| Area | master layout | feature_master layout |
|---|---|---|
| Top-level kernel/runtime | `kernel/`, `runtime/` at root | moved under `sw/kernel/`, `sw/runtime/` |
| Hardware config | `hw/rtl/VX_config.vh`, `hw/rtl/VX_types.vh` | `VX_config.toml`, `VX_types.toml` at repo root |
| simx execution | monolithic `execute.cpp` + `func_unit.cpp` + `emulator.cpp` | split per-unit: `alu_unit.cpp`, `fpu_unit.cpp`, `sfu_unit.cpp`, `csr_unit.cpp`, `lsu_unit.cpp`, `wctl_unit.cpp`, `barrier_unit.cpp`, … |
| simx memory hierarchy | `sim/simx/local_mem.{cpp,h}`, `cache_sim.{cpp,h}`, `mem_sim.{cpp,h}`, `mem_coalescer.{cpp,h}` at `sim/simx/` | renamed and moved under `sim/simx/mem/`: `local_mem.*`, `local_mem_switch.*`, `cache.*` (was `cache_sim`), `memory.*` (was `mem_sim`), `cache_cluster.*`, `lsu_mem_adapter.*`, `mem_block_pool.h` |
| simx scheduling | `pipeline.h`, `dispatcher.{cpp,h}`, `scoreboard.h`, `ibuffer.h` | `scheduler.{cpp,h}`, `sequencer.{cpp,h}`, `scoreboard.{cpp,h}`, `dispatcher.{cpp,h}` |
| New simx subsystems | none | `sim/simx/dxa/`, `sim/simx/kmu/`, `sim/simx/tcu/` |
| New RTL subsystems | none | `hw/rtl/dxa/`, `hw/rtl/kmu/`, plus split `VX_kmu.sv` |

A merge in either direction would conflict on every PR that touches a
simx, kernel, or runtime file. The conflict-resolution work would be
indistinguishable from a manual replay, but with a useless merge commit
on top and the original PR boundaries lost.

We replay PR-by-PR instead.

---

## 3. Inventory of incoming changes

35 commits land on master between 2026-04-19 and 2026-04-24. They group
into 15 upstream PRs (external + tinebp), 1 already-merged-back PR
(#328), and a tail of direct tinebp commits.

### 3.1 External-contributor PRs

| PR | Title | Author | Files (paths as on master) |
|---|---|---|---|
| #267 | fix: resolve memory access violation in `io_addr` test | Dingyi Zhao `<dingyizhao.zdy@outlook.com>` | `tests/regression/io_addr/main.cpp` (+1/-1) |
| #282 | OpenMPI for SimX (mpi_vecadd, mpi_blocked_sgemm, mpi_conv3, mpi_diverge, mpi_dotproduct, mpi_neighbor_a2a_conv3, mpi_put_dotproduct, mpi_sgemm) | Rahul Raj D N `<rahul7rajdn@gmail.com>` | new `tests/regression/mpi_*/`, `tests/regression/Makefile`, `tests/regression/common.mk`, `ci/blackbox.sh`, `miscs/apptainer/vortex.def` (+2742/-3 across 37 files) |
| #289 | Apptainer-based CI pipeline | Udit Subramanya `<usubramanya91@gmail.com>` | new `.github/workflows/apptainer-ci.yml` (+292) |
| #297 | AXI burst mode Fixed→Incr for Vivado SmartConnect | Rosi Carannante `<rosi.carannante@studenti.unina.it>` | `hw/rtl/libs/VX_axi_adapter.sv` (+2/-2) |
| #298 | SimX-SST integration (no memory integration) | Jagadheesvaran T S `<jagadheesvaran.t.s@gmail.com>` + tinebp fixups | `sim/simx/processor.{cpp,h}`, `sim/simx/processor_impl.h`, new `sim/simx/VortexGPGPU.{cpp,h}`, new `sim/simx/vortex_simulator.{cpp,h}`, `sim/simx/Makefile`, `ci/sst_*.py`, `ci/sst_install.sh.in`, `ci/regression.sh.in`, `.github/workflows/ci.yml` (+462/-3 across 15 files) |
| #306 | Update microarchitecture.md | jaredbfrost `<73719801+jaredbfrost@users.noreply.github.com>` | `docs/microarchitecture.md` (+2/-2) |
| #310 | CopyBuf support (OpenCL test + runtime memcpy device-match) | talubik `<rustem149849@gmail.com>` + tinebp fixup | `runtime/include/vortex.h`, `runtime/common/callbacks.{h,inc}`, `runtime/{opae,rtlsim,simx}/vortex.cpp`, `runtime/opae/driver.{cpp,h}`, new `tests/opencl/copybuf/`, `tests/opencl/Makefile`, `sim/xrtsim/xrt_sim.h` (+424/-1 across 22 files) |
| #320 | Fix srai decode + arith test | cassuto `<diyer175@hotmail.com>` | decode logic + new `tests/regression/arith/` (+243/-1 across 5 files) |
| #321 | Fix vx_spawn cooperative-thread bookkeeping | talubik `<rustem149849@gmail.com>` | `kernel/src/vx_spawn.c` (+12/-2) |
| #324 | Add cache size configuration examples to docs | Nikita Churin `<churin.nick2006@gmail.com>` | `docs/simulation.md` (+47) |
| #326 | Rename `__assert` → `__vortex_assert` (fixes debug builds) | Nara Díaz Viñolas `<nara.diaz@bsc.es>` | `sim/common/simobject.h`, `sim/common/util.h` (+14/-14) |
| #327 | Fix local_mem in simx | talubik `<rustem149849@gmail.com>` | `sim/simx/local_mem.cpp` (+3/-4) |
| #330 | platforms.mk: add U50 FPGA | Rahul Raj D N | `hw/syn/xilinx/xrt/platforms.mk` (+1) |
| #338 | Fix data race in bss (move bss zero-init from `vx_start.S` to runtime) | talubik | `kernel/src/vx_start.S`, `runtime/stub/utils.cpp` (+9/-6) |

### 3.2 PR already round-tripped

| PR | Note |
|---|---|
| #328 | Was merged back from `vortexgpgpu/tinebp-patch-2` (Verilator Focal download). The fix is **already present** in `ci/toolchain_install.sh.in:119` on the current branch (`"ubuntu/focal") parts=$(eval echo {a..c}) ;;`). Skip. |

### 3.3 Direct tinebp commits on master (no PR boundary)

| SHA | Subject | Files |
|---|---|---|
| `f00bb142` | ci: fix toolchain scripts for new Verilator + add sta tool | `ci/toolchain_install.sh.in`, `ci/toolchain_prebuilt.sh.in` |
| `9b5e6645` | refactor(dtm): relocate debug stack, drop unrelated TF32, add coverage | 18 files across `sim/simx/`, `hw/rtl/`, tests |
| `a537f2a5` | Merge PR #309 (Use debug mode with word) | 23 files; touches `sim/simx/remote_bitbang.h`, `sim/simx/socket.h`, `sim/simx/tcu/tensor_unit.cpp`, `vortex.cfg` |
| `0762760b` | SST script update | `ci/sst_test_vortex_*.py` |
| `ca2a1620` | simx source tree restructuring to support submodules | `sim/simx/emulator.h`, `sim/simx/sst/vortex_gpgpu.{cpp,h}`, `sim/simx/sst/vortex_simulator.cpp` |
| `5c4adf50` | simx source tree restructuring for submodules | rename-only: moves `vopc_unit.*`, `voperands.*` etc. into `sim/simx/vpu/` |
| `7793aa07` | PR regression fixes | `sim/{rtlsim,simx}/main.cpp`, `sim/xrtsim/Makefile`, `tests/kernel/common.mk`, … |
| `1eb9f74f` | ci: gate `mpi_*` regression tests behind `MPI=1` | regression makefiles |
| `761f8ad1` | ci: add mpi test coverage for regression suite | `.github/workflows/ci.yml`, `ci/install_dependencies.sh`, `ci/regression.sh.in`, `tests/regression/Makefile` |
| `72112f39` | fix: relax `mpi_blocked_sgemm` ULP tolerance for Cannon's algorithm | `tests/regression/mpi_blocked_sgemm/main.cpp` |

(Plus housekeeping: `5a9d8d2f`, `305c201a`, `c1099215`, `3c8a7dd1` —
update supported OS, drop Ubuntu 22.04, build update, build fix. These
fold into the toolchain/CI replay.)

---

## 4. Compatibility classification

Each incoming change is bucketed by how it interacts with the
feature_master reorg.

### 4.1 **CLEAN** — straight cherry-pick, conflicts unlikely

These touch files that exist at the same path on both branches and do
not collide with the simx_v3 / sw-reorg / TOML-config refactors.

- **PR #267** — `tests/regression/io_addr/main.cpp` exists at the same
  path. Pure off-by-one fix. **Risk: low.**
- **PR #289** — new file `.github/workflows/apptainer-ci.yml`. **Risk: low.**
- **PR #297** — `hw/rtl/libs/VX_axi_adapter.sv` is a leaf adapter,
  unaffected by the cluster/socket/gpu_pkg reorg. **Risk: low.**
- **PR #306** — docs. **Risk: low.**
- **PR #320** — decoder fix in current branch's `sim/simx/decode.cpp`
  + new `tests/regression/arith/` folder. Decoder file exists on both
  sides; the fix is one case-statement entry. **Risk: low.**
- **PR #324** — docs. **Risk: low.**
- **PR #330** — one-line addition to `hw/syn/xilinx/xrt/platforms.mk`.
  **Risk: low.**

### 4.2 **PATH-RENAME** — cherry-pick with `--no-commit`, then re-stage to renamed location

These touch files that exist on both branches but at a different path
because of the `sw/` reorg.

- **PR #321** (vx_spawn) — `kernel/src/vx_spawn.c` on master,
  `sw/kernel/src/vx_spawn.c` on current. Apply with
  `git cherry-pick --no-commit`, then `git mv`/manual-stage and verify
  the spawn-bookkeeping diff still matches the surrounding code (the
  current branch may have already evolved this file for CTA work — see
  `sw/kernel/src/vx_spawn.c` and the recent `cta_dispatcher` commits;
  apply manually if the surrounding context has shifted). **Risk: medium.**
- **PR #338** (bss data race) — `kernel/src/vx_start.S`,
  `runtime/stub/utils.cpp` on master → `sw/kernel/src/vx_start.S`,
  `sw/runtime/stub/utils.cpp` on current. **Risk: medium.**
- **PR #310** (CopyBuf) — paths `runtime/...` → `sw/runtime/...`,
  `tests/opencl/copybuf/` is new (clean), `runtime/include/vortex.h`
  is one of the few touched on the current branch's `sw/runtime/include/`
  too — review the signature for `vx_copy_buffer` against the existing
  callbacks table. **Risk: medium.**

### 4.3 **REWRITE** — discard the upstream patch, re-apply the *intent* on current

These touch files that have been renamed or rewritten such that the
upstream diff cannot be applied mechanically. The fix must be ported by
hand, with the original author preserved via `--author` and the source
SHA cited in the commit body.

- **PR #326** (`__assert` → `__vortex_assert`) — the upstream patch
  edits `sim/common/simobject.h` and `sim/common/util.h`. On current,
  `sim/common/util.h` does not exist (utility code reorganized) and
  `sim/common/simobject.h` still uses `__assert(...)`
  ([sim/common/simobject.h:296](../../sim/common/simobject.h#L296) and
  ~9 more sites). The rename is the right fix (motivation: collision
  with the C-stdlib `__assert` symbol on debug builds), but the patch
  must be re-applied to the current set of `__assert` call sites and the
  macro definition must be hunted in its current home (search
  `sim/common/`). Re-author to Nara Díaz Viñolas, body
  `Original PR #326 (commit 8d25736d). Re-applied for sw-reorg layout.`
  **Risk: low (mechanical) once the macro definition is located.**
- **PR #327** (local_mem in simx) — upstream patches
  `sim/simx/local_mem.cpp`. Current branch's
  `sim/simx/mem/local_mem.cpp` is a substantial rewrite tied to the
  TLM data-carrying redesign (see Phase 5 of
  [simx_v3_proposal.md](simx_v3_proposal.md)). Read the upstream 3-line
  fix to understand *what* bug is being fixed, then verify it is either
  (a) already absent in the rewrite or (b) port the equivalent
  semantic. **Risk: medium — must read the upstream bug before deciding.**
- **PR #298** (SimX-SST integration) — upstream introduces
  `sim/simx/VortexGPGPU.{cpp,h}` and `sim/simx/vortex_simulator.{cpp,h}`
  and edits `processor.cpp/h`. The proposal originally assumed current
  branch already had `sim/simx/sst/` from `ca2a1620`; **that assumption
  was wrong** — `ca2a1620` is on master, not on this branch. See §10
  for the actual disposition.

### 4.4 **MERGE-AS-IS** — large, self-contained, low-risk

- **PR #282** (OpenMPI suite) — adds 8 new `mpi_*` regression
  directories and touches `tests/regression/Makefile`,
  `tests/regression/common.mk`, `ci/blackbox.sh`,
  `miscs/apptainer/vortex.def`. The new directories are net-new; the
  Makefile/common.mk additions need to be diffed against current's
  versions and merged by hand if the structure has changed. The
  follow-on tinebp commits (`72112f39`, `761f8ad1`, `1eb9f74f` — ULP
  tolerance, MPI gating, CI coverage) belong with this batch and
  should be replayed in the same merge unit. **Risk: medium-high
  (volume).**

### 4.5 **SKIP** — already on the current branch in equivalent form

- **PR #328** — Verilator Focal download. Already at
  [ci/toolchain_install.sh.in:119](../../ci/toolchain_install.sh.in#L119).
- `5c4adf50` (vpu reorg), `ca2a1620` (sst reorg) — current branch's
  reorg supersedes these; the equivalent layout already exists.
- `5a9d8d2f` / `305c201a` (drop Ubuntu 22.04) — fold into the
  toolchain replay if not already present; check `.github/workflows/ci.yml`.

### 4.6 **DEFER** — out of scope for this merge

- `9b5e6645` (DTM relocate + drop TF32 + coverage) — this is
  a **structural change** from tinebp on master that affects
  `sim/simx/emulator.{h,cpp}`, `sim/simx/execute.cpp`,
  `sim/simx/tcu/tensor_unit.cpp`. The simx_v3 line has its own DTM
  trajectory (debug-stack location decisions are bound up with the
  `Emulator` decomposition in
  [simx_v3_proposal.md §2.6](simx_v3_proposal.md)). Defer to a follow-up
  proposal; record the intent only.
- `a537f2a5` (Merge PR #309 — debug-mode-with-word + 23-file deltas)
  — same reason: large, cross-cuts into TCU/socket/remote_bitbang
  which are mid-refactor on this line. Defer.

---

## 5. Strategy

### 5.1 Approach (chosen)

**Per-PR replay, in dependency order, with `cherry-pick -x` where the
patch is mechanical and `commit --author` + body-cited SHA where it is
not.**

For each PR:

```
# Mechanical case (CLEAN / PATH-RENAME)
git cherry-pick -x <inner-PR-commit-sha> [<inner-PR-commit-sha-2> ...]
# (-x appends "(cherry picked from commit ...)" — gives traceability
# to the master SHA without pretending it's a merge.)

# Rewrite case (REWRITE / MERGE-AS-IS conflict resolution)
git apply --3way <patch>          # try first
# resolve, then:
git commit \
  --author="Nara Díaz Viñolas <nara.diaz@bsc.es>" \
  -m "rename __assert to __vortex_assert (fixes debug builds)

Original PR #326 (commit 8d25736d). Re-applied for sw-reorg layout."
```

Each PR closes with one final tinebp commit on top:
`Merge PR #<N>: <title>` — empty diff, present only to mark the PR
boundary in `git log --first-parent` output. This recreates the
"one PR ↔ one merge marker" property without needing a real
2-parent merge (which would drag in unrelated master history).

### 5.2 Why not `git merge --no-ff <PR-branch>`

Would record a real 2-parent merge and a pristine branch graph — but
each per-PR merge would also drag in any unrelated master commits that
preceded it on the master timeline (the merge-base for each PR is
2025-07-11, not the prior PR). The result is either (a) a series of
independent fast-forwards from one common base, all conflicting against
each other and against the simx_v3 reorg, or (b) one giant merge of
master HEAD that defeats the per-PR review goal.

### 5.3 Why not pure rebase

`git rebase --onto feature_master <base> <master>` would also preserve
authorship, but loses the PR-boundary structure (everything becomes a
flat commit list) and triggers all 35 conflicts in one sitting with no
incremental review.

### 5.4 Order of operations

Replay in this order — earlier items unblock later ones, and the
high-risk REWRITEs go last so the branch is in a known state before
they start.

1. **Round 1 — CLEAN (no risk):** #267, #289, #297, #306, #320, #324, #330
2. **Round 2 — PATH-RENAME:** #321, #338, #310
3. **Round 3 — MERGE-AS-IS:** #282 + tinebp tail (`72112f39`, `761f8ad1`, `1eb9f74f`, `f00bb142`, `7793aa07` build/CI)
4. **Round 4 — REWRITE:** #326, #327, then PR #298 verification (skip-with-credit)
5. **Round 5 — Defer-list documentation:** add a note in
   `docs/proposals/master_merge_v3_proposal.md` (this file) recording
   what was deferred and why, so the next merge cycle knows where to
   pick up.

Each round ends with a build-and-smoke checkpoint (§7).

---

## 6. Authorship / history mechanics

### 6.1 Cherry-pick with traceability

```
git cherry-pick -x <sha>
```

`-x` appends `(cherry picked from commit <upstream-sha>)` to the body.
This is the canonical way to keep a back-pointer without lying about
the parent graph. The `author` field is preserved automatically; the
`committer` becomes the local user (`tinebp`), which is correct.

### 6.2 Re-apply with `--author`

For REWRITE cases the upstream commit is functionally re-implemented,
so a cherry-pick is not honest. Use:

```
git commit \
  --author="<original-author> <original-email>" \
  --date=<original-author-date> \
  -m "<original subject>

Original PR #<N> (commit <upstream-sha>). Re-applied for <reason>.
Co-Authored-By: <original-author> <original-email>"
```

`--author` and `--date` set the commit-level author and author-date;
`Co-Authored-By:` is a redundant safety net visible to GitHub's
contribution graph.

### 6.3 PR-boundary markers

After each PR's commits land, an empty marker commit closes the unit:

```
git commit --allow-empty -m "Merge PR #<N>: <title> (replayed from master)"
```

Result: `git log --first-parent` shows one line per PR, exactly mirroring
the master-side narrative. `git log` shows every original
contributor commit with its original author, in order.

---

## 7. Validation

After each round (§5.4):

1. **Build**, only inside `vortex_v3/feature_master/build_test32`:
   ```
   ../configure --tooldir=/opt && make -s
   ```
2. **Smoke run**: the standard simx + rtlsim smoke (`vecadd`,
   `dotproduct`, plus one TCU/WGMMA test if Round 4 has touched the
   tensor path).
3. **Targeted regressions per round:**
   - Round 1: run `tests/regression/io_addr` (#267), `tests/regression/arith` (#320).
   - Round 2: run `vx_spawn`-touching tests (CTA spawn coverage), bss-init coverage.
   - Round 3: `MPI=1 make -C tests/regression mpi_vecadd` and
     `mpi_blocked_sgemm`.
   - Round 4: full simx debug build (the `__assert` rename is the
     fix-trigger — debug build must compile clean), local_mem
     coverage in simx.
4. **Trace diff** vs. pre-merge `feature_master` HEAD on a baseline
   kernel (e.g. `vecadd`): expect no change (the merged PRs do not
   touch core simx datapaths). Any diff is a regression and must be
   triaged before proceeding to the next round.

A round that fails validation is reverted as a unit and re-attempted;
no half-merged PRs survive a checkpoint.

---

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | A REWRITE replay accidentally re-introduces a v3-forbidden pattern (e.g. a `core->mem_read` from a backported local_mem fix). | §1 constraint #3 plus a grep gate in the validation script: `git diff <pre>..<post> -- sim/simx/ \| grep -E 'core->mem_(read\|write)'` must be empty for any commit not on the v3-deferred list. |
| R2 | PR #298 is *not* equivalent to current's `sim/simx/sst/`, and we lose CI coverage by skipping it. | Round 4 starts by diffing `ci/sst_*.py` and `ci/sst_install.sh.in` between master and current; if current is missing test logic, port it as a credited rewrite (Jagadheesvaran). |
| R3 | The OpenMPI tests (#282) depend on toolchain pieces (apptainer recipe `miscs/apptainer/vortex.def`) that may collide with current's apptainer setup. | Round 3 explicitly diffs `miscs/apptainer/vortex.def` before applying; conflicts are resolved by hand (treat the apptainer recipe as a single hand-edited file, not a cherry-pick target). |
| R4 | The "PR-boundary marker" empty commits clutter `git log` for future bisect users. | They are tagged in the body (`replayed from master`) and remain empty — `git bisect` skips them automatically when bisecting by behavior change. |
| R5 | The deferred PRs (`9b5e6645`, `a537f2a5`) silently bit-rot. | §5.4 step 5 records them in this proposal under §4.6 with their SHAs. Next merge cycle starts by reading this section. |

---

## 9. Out of scope

- Pushing any of this back to upstream master.
- Merging PRs that landed on master *before* 2026-04-19 (older PRs are
  presumed already absorbed; verify via `git log HEAD..FETCH_HEAD
  --before=2026-04-19` if doubted).
- Resolving the `VX_config.vh` / `VX_types.vh` ↔ `VX_config.toml` /
  `VX_types.toml` migration in either direction. The merge does not
  touch those files; if a future master PR needs the .vh form, it
  becomes a REWRITE under the policy in §4.3.
- The DTM / TF32 / debug-mode reorg (`9b5e6645`, `a537f2a5`). Tracked
  separately; will need its own proposal once the simx_v3 emulator
  decomposition lands.

---

## 10. Outcomes (post-execution, 2026-05-03)

The merge ran on 2026-05-03. New rule applied throughout: **keep ours
(HEAD) on every conflict unless theirs is a bug fix.** Recorded as
[`feedback_keep_ours_in_merge.md`](../../../../.claude/projects/-home-blaisetine-dev/memory/feedback_keep_ours_in_merge.md).
This rule overrides the per-PR risk table where they conflict — in
particular it changed the disposition of PR #298 (now "not adopted",
not "verify equivalence").

### 10.1 What landed

| Round | PRs | Author-preserved commits | Markers / fixups |
|---|---|---|---|
| 1 — CLEAN | #267, #289, #297, #306, #320, #324, #330 | 8 (Dingyi Zhao, Rosilena, jaredbfrost, Nikita Churin, rahul7rajdn, Udit Subramanya, cassuto ×2) | 7 markers + 1 wire-up (`arith` into regression Makefile — upstream forgot it) |
| 2 — PATH-RENAME | #321, #338, #310 | 4 (talubik ×2 vx_spawn, talubik bss, talubik copybuf) + 1 tinebp fixup (memmove direction) | 3 markers + 2 fixups (RAM::get 2-arg signature; copybuf run-target gate) |
| 3 — MERGE-AS-IS | #282 + tinebp MPI/CI tail | 3 Rahul Raj D N (OpenMPI suite) + 5 tinebp tail (gate MPI=1, ULP, CI MPI integration, Verilator toolchain, PR regression fixes, vxbin build fix) | 1 marker + 1 closeout + 1 fixup (vxbin path) |
| 4 — REWRITE | #326, #327, #298 | 1 (Nara Díaz Viñolas, full rewrite for sw-reorg) | 3 markers (1 absorbed, 1 not-adopted, 1 closeout) |

Total: 16 author-preserved external/contributor commits + 1 author-preserved
re-applied commit, plus tinebp's tail and markers. All Round 1 + Round 2
+ Round 3 smoke tests pass after every round (io_addr, arith, vecadd,
mpi_vecadd MPI=1 NP=2).

### 10.2 Disposition deltas vs. the original §4 plan

| PR | Plan said | What actually happened | Why |
|---|---|---|---|
| #320 (srai) | CLEAN cherry-pick | Cherry-pick conflicted (decoder shape diverged); re-applied as a 1-line manual edit with cassuto preserved as author. | Current branch's `Decoder::decode` uses single-line `set_op_type`; upstream uses block-form `setOpType`. The bug fix (`==` → `&`) is the same; the surrounding code is not. |
| #320 follow-up | (n/a) | Added `arith` to `tests/regression/Makefile` `all`/`run-simx`/`run-rtlsim`/`clean`. | Upstream PR #320 added the test directory but never wired it into the parent Makefile, so `make all` would skip it. Wired it in as a tinebp follow-up. |
| #310 (CopyBuf) | PATH-RENAME, low risk | PATH-RENAME with 8 conflict files; OpenCL test segfaults at runtime. | Conflicts were the predictable adjacent-add pattern (kept HEAD's reshaped `start()` signature; added theirs' `copy_dev_to_dev`). The OpenCL `clEnqueueCopyBuffer` path crashes inside the pre-built `/opt/pocl` because pocl-vortex was compiled before PR #310 added the buffer-copy opcode — its command-handler table has a NULL entry. The runtime C API (`vx_copy_dev_to_dev`) works; only the OpenCL path is broken. Disabled `copybuf run-*` in `tests/opencl/Makefile` with a TODO referencing PR #310. |
| #298 (SST) | "Already integrated, skip-with-credit" | **Not adopted.** | Original §4.3 was based on a wrong assumption (that `ca2a1620` was on this branch). It is not. Adopting PR #298 would be a *net-new feature* on this line, which violates the "keep ours unless bug fix" rule. The mem_backend abstraction also conflicts with the TLM redesign in [simx_v3_proposal.md](simx_v3_proposal.md) Phase 5. SST integration on this line, if/when needed, should be designed on top of TLM rather than retrofitted. |
| #327 (local_mem) | "Read upstream bug, port equivalent" | **Already absorbed.** | Current branch's `sim/simx/mem/local_mem.cpp` already uses `log2ceil(config.capacity)` (the corrected mask width). Only the variable name (`line_bits_` vs `addr_bits_`) differs — cosmetic, kept ours per the rule. talubik credited in the marker commit. |
| #326 (`__assert` rename) | REWRITE, low risk | REWRITE landed, but the patch surface was larger than expected: not just `sim/common/simobject.h` but also `sw/common/ringqueue.h` and `sw/common/smallfunc.h` use the macro. | Same logic as the proposal said; just more files to edit. Single re-applied commit by Nara Díaz Viñolas via `--author=` + `Co-Authored-By:`. |
| `761f8ad1` (CI MPI integration) | "Take it" | Took it. Conflict in `ci/regression.sh.in` introduced `mpi()` and `sst_tests()` shell functions; **kept `mpi()` only**, dropped `sst_tests()` (consistent with the §10.2 PR #298 decision), and **path-translated** `runtime/simx` → `sw/runtime/simx`. |
| `1eb9f74f` (gate MPI=1) | "Take it" | Took it. Original conflict was destructive (theirs wanted to *remove* my mpi_* lines from `all`/`run-simx` and put them in `ifeq($(MPI),1)` block at end). I accepted theirs because the gating IS the bug fix — running `make all` without OpenMPI installed would otherwise fail. |
| `761f8ad1` follow-up | (n/a) | The `tests/regression/Makefile` final shape replaced `1eb9f74f`'s inline `ifeq($(MPI),1)` block with separate `all-mpi:` / `run-simx-mpi:` targets. Cleaner design from `761f8ad1`; this is what stuck. |
| `7793aa07` (PR regression fixes) | (not classified — folded into Round 3 tail) | Took the additive bits (`RAM::loadVxImage`, `tensor_cfg.h`, vxbin/link-script changes). **Kept ours** on the Makefile XCONFIGS-vs-CONFIGS rename conflicts (cosmetic, not bug fix) and the AFU `vx_reset` reshape (RTL design diverged on this line). |
| `7793aa07` follow-up | (n/a) | `tests/kernel/common.mk` referenced the old `kernel/scripts/vxbin.py` path. Fixed up to `sw/kernel/scripts/vxbin.py` so `make` would not fail. |
| #328, `5a9d8d2f`, `305c201a`, `c1099215`, `3c8a7dd1` | SKIP / housekeeping | #328 is already present (focal Verilator). `5a9d8d2f` (README OS bump) and `305c201a` (drop ubuntu-22.04 from CI) are housekeeping that we don't need (we're already on `[ubuntu-24.04]` only). `c1099215` renames `sst_tests` → `sst` in CI (we don't have either). `3c8a7dd1` is the vxbin `_end` symbol fix and **was applied** as a real bug fix. |

### 10.3 Known issues post-merge

1. **OpenCL copybuf test crashes pocl.** Tracked in
   `tests/opencl/Makefile` with a TODO. Re-enable after rebuilding
   `/opt/pocl` against the new runtime ABI (PR #310 added the
   buffer-copy opcode; pre-built pocl-vortex doesn't dispatch it).
2. **MPI tests are gated behind `MPI=1`.** `make all` and `make
   run-simx` in `tests/regression` do **not** include `mpi_*` —
   use `make all-mpi MPI=1 NP=4` and `make run-simx-mpi MPI=1 NP=4`.
   This matches CI (`./ci/regression.sh --mpi`).
3. **PR #298 (SST) is intentionally absent.** Documented in §10.2.
   SST users following the upstream master will not find their
   integration on this line.
4. **No XCONFIGS → CONFIGS rename was taken** (kept the per-line
   divergence). If a future upstream PR uses `CONFIGS` for FPNEW
   gating, that PR will need REWRITE treatment.

### 10.4 Outstanding (still deferred)

Same as §4.6, unchanged:
- `9b5e6645` — DTM relocate + drop TF32 + coverage. Cross-cuts the
  Emulator decomposition; needs its own proposal.
- `a537f2a5` — Merge PR #309 (debug mode with word). 23-file delta
  spanning TCU/socket/remote_bitbang. Needs its own proposal.

These two represent the only remaining catch-up debt against master
HEAD (`f00bb142`, 2026-04-24) for the past-2-weeks window.
