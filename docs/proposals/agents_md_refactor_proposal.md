# AGENTS.md Refactor: Central Knowledge Hub for Code Agents

**Status:** Draft for review
**Scope:** `vortex_ci/AGENTS.md` and the `docs/` tree it points to
**Audience:** Human contributors and AI coding agents (Claude, Codex, Gemini, ...)

---

## 1. Motivation

The current [AGENTS.md](../../AGENTS.md) is a useful onboarding sheet, but it has drifted into a **build/test recipe document**, not a *foundation-of-knowledge document*. It tells an agent how to invoke `blackbox.sh`, but it does not capture the rules that have been learned the hard way over many sessions — for example:

- "Always build out-of-tree under `build/`, never in the source root."
- "Install toolchain into `$HOME/tools`, never `/opt`."
- "After edits to source Makefiles, re-run `../configure` from `build/` before the change takes effect."
- "RTL coverage goes through `xrt`, not `rtlsim`."
- "`RTL_PKGS` is for `VX_*_pkg.sv` only — Verilog interfaces are illegal there."
- "When in doubt, pick the solution that aligns with how a real NVIDIA GPU/driver works."
- "Don't pause multi-phase feature work to ask about commits; only ask once the entire feature is end-to-end tested."
- "Code comments lead with *why*, not *what*; never reference the current task, PR, or caller."

These rules currently live in three uncomfortable places:

1. **A few maintainers' heads.**
2. **Per-agent memory files** (e.g., Claude's `~/.claude/.../memory/feedback_*.md`).
3. **Scattered hints inside individual docs** (`coding_guidelines_cpp.md`, `debugging.md`, etc.) — often partial or stale.

The cost of this fragmentation:

- New contributors (human or AI) repeat the same mistakes, and maintainers re-explain the same rules.
- Each AI agent re-discovers conventions independently, so behavior is inconsistent across Claude / Codex / Gemini.
- Rules learned across `feature_gfx`, `feature_cp`, `feature_gem5`, `feature_vulkan`, `feature_hip` branches are not back-propagated into a place where future work can find them.

### Goal

Promote **AGENTS.md** from "build cheat-sheet" to the **canonical entry point** that every code agent reads before touching the Vortex codebase. AGENTS.md itself stays small and stable; it acts as an **index with strong opinions**, deferring depth to a curated set of topic docs under `docs/`.

---

## 2. Design Principles

1. **Single source of truth, layered.** AGENTS.md links to one — and only one — authoritative doc per topic. No duplication of substantive content across docs.
2. **Rules over recipes.** AGENTS.md captures *invariants* ("install toolchain to `$HOME/tools`"); recipes ("here is a 30-line `blackbox.sh` invocation") live in the topic doc.
3. **Each rule has a "why."** A rule without a rationale rots. Every entry includes the motivation (often a real incident or a design principle).
4. **Audience-symmetric.** What humans need is what agents need. We do not maintain two parallel sets of guidelines.
5. **Living document.** AGENTS.md and the docs it points to are versioned alongside the code. Branch-specific addenda are *forbidden* — branch knowledge gets distilled and merged into the trunk doc set when the branch lands.
6. **Reading budget aware.** The AGENTS.md body itself should fit comfortably in an agent's initial context window (target: ≤ 300 lines, ~10–12 KB). Depth flows through links.

---

## 3. Current State Audit

### 3.1 What `AGENTS.md` does today (226 lines)

| Section | Verdict |
|---|---|
| Documentation Map | Keep — but expand and reorganize by topic, not by filename |
| Build Directory Setup | Move bulk to `install_vortex.md`; keep only the "out-of-tree, `$HOME/tools`, re-configure after Makefile edits" *rules* |
| Common Gotchas | Move to `debugging.md` / `testing.md`; keep the *one-liner* rules in AGENTS.md |
| Testing & Debugging | Move recipes to `testing.md`/`debugging.md`; AGENTS.md keeps the 120s-timeout rule, the `--rebuild` rule, the `CONFIGS` matching rule |
| Configuring architecture parameters | Move to `simulation.md` or a new `configuration.md` |
| Graphics extensions | Move to `microarchitecture.md` or a dedicated `extensions.md`; AGENTS.md keeps a one-line pointer |
| General Notes | Expand into proper "Design Rules" + "Commit Rules" + "Comment Rules" sections — these are the highest-value bits and are currently underdeveloped |

### 3.2 Rules currently *missing* from AGENTS.md (sourced from per-agent memory)

Tagged by category — these are the rules we want formalized into the new structure:

**Build & toolchain**
- Out-of-tree builds only (`build/`); never `cmake`/`make` in source root.
- Toolchain prefix is `$HOME/tools`, never `/opt` or `/usr/local`.
- `configure` copies source Makefiles into `build/`; edits to source require re-`../configure`.
- ccache can serve stale objects on `fmt` version mismatches → retry with `CCACHE_DISABLE=1` before deep-diving.

**Testing & verification**
- All test invocations capped at 120 s (`timeout 120` or `Bash` `timeout: 120000`).
- For RTL coverage, name `xrt` as the path; `rtlsim` bypasses the AFU surface.
- `blackbox.sh` rebuilds the *driver* but not the *app* — match `CONFIGS` on both sides.

**Design**
- Prefer root-cause fixes over patches. If a patch is unavoidable, label it as such and file a follow-up.
- For ambiguous design questions, pick the solution a real NVIDIA GPU/driver would use.
- Runtime API (`vortex2.h`) stays minimal (Vulkan/CUDA/Metal style). Push complexity to translators, never into the core runtime.

**RTL**
- `RTL_PKGS` is for `VX_*_pkg.sv` only. Verilog interfaces are illegal there — fix discovery via include paths / file naming.
- Library modules under `hw/rtl/libs/` default to `TRACING_OFF`; toggle with `TRACING_ALL` for a full trace.

**Terminology**
- "RTL CP" = `hw/rtl/cp/*.sv` hardware. "Emulation CP" / "Simulation CP" = `sim/common/cmd_processor.cpp` C++ model. Never "host-side CP" or "software CP".

**Documentation**
- Design/migration proposals belong in `docs/proposals/`, not in build dirs or the repo root.

**Commits & branching** (project-policy section — see §5 for handling of public-vs-internal split)
- Substantial, testable feature units per commit; no skeletons, no WIP, no micro-commits.
- Feature branches receive *direct commits*; PRs are reserved for landing onto trunk.
- Don't pause multi-phase work to ask about commits; only ask once the feature is end-to-end tested.
- When replaying upstream onto a divergent branch, default to keeping `HEAD`; take `theirs` only for clear bug fixes.

**Code comments**
- Lead with *why*, not *what*. Identifiers carry the *what*.
- No references to the current task / PR / caller. Those belong in the commit message and rot in code.
- No multi-paragraph docstrings or multi-line comment blocks for trivial code.

---

## 4. Proposed Structure

### 4.1 New AGENTS.md skeleton

```
# AGENTS Guide for Vortex GPGPU Development

[1-paragraph orientation: what Vortex is, what an agent should expect]

## 0. Read This First (Foundation Rules)
[The 6–10 highest-leverage rules an agent must internalize before any change.
 Each line: one-sentence rule + link to authoritative doc.]

## 1. Documentation Map (by topic)
[Grouped: Setup · Codebase · Coding · Simulation · Debug · Test · Design · Process]

## 2. Build & Toolchain Rules
[Pointers + 5–10 line invariants. Recipes live in install_vortex.md.]

## 3. Testing & Verification Rules
[Pointers + invariants. Recipes live in testing.md / debugging.md.]

## 4. Design & Architecture Rules
[NVIDIA alignment, root-cause fixes, runtime minimalism. Pointers to
 microarchitecture.md and active proposals.]

## 5. Coding Conventions
[Pointers to coding_guidelines_cpp.md / coding_guidelines_verilog.md +
 cross-cutting comment rules.]

## 6. Commit & Branching Rules
[Pointers to contributing.md. Project-policy section — see §5 below.]

## 7. Living Document Policy
[How and when to update AGENTS.md and the docs it points to.]
```

### 4.2 Topic doc updates

| Doc | Action | What changes |
|---|---|---|
| `docs/index.md` | **Update** | Mirror AGENTS.md's topic grouping; remove now-redundant pointer-to-pointer entries |
| `docs/codebase.md` | **Update** | Refresh tree to current state (`hw/rtl/cp/`, `sim/common/cmd_processor.cpp`, etc.); reflect graphics units; align terminology |
| `docs/install_vortex.md` | **Update** | Absorb the "Build Directory Setup" block from AGENTS.md; add `$HOME/tools` rule and re-`configure` rule explicitly |
| `docs/coding_guidelines_cpp.md` | **Update** | Add cross-cutting "comments lead with why, not what" rule; add "no task/PR/caller references" rule |
| `docs/coding_guidelines_verilog.md` | **Update** | Add `RTL_PKGS` constraint; add `TRACING_ON/OFF/ALL` convention for `libs/` modules |
| `docs/testing.md` | **Update** | Absorb `--rebuild` rule, `CONFIGS`-matching gotcha, 120 s cap; document the regression matrix entry point (`ci/regression.sh`) |
| `docs/debugging.md` | **Update** | Add the ccache-stale-objects gotcha; clarify "rtlsim vs xrt" coverage roles |
| `docs/simulation.md` | **Update** | Move the `CONFIGS` / blackbox override reference here from AGENTS.md |
| `docs/contributing.md` | **Major rewrite** | Currently public-fork/PR-flow only. Add a section on feature-branch direct-commit workflow used internally, and the "substantial feature per commit" rule. **See §5 below.** |
| `docs/design_rules.md` | **NEW** | Captures the design-philosophy rules: NVIDIA alignment, root-cause fixes, runtime minimalism, CP terminology |
| `docs/extensions.md` *(optional)* | **NEW** | Pull the TEX/RASTER/OM section out of AGENTS.md. Could alternatively fold into `microarchitecture.md`. |
| `docs/proposals/README.md` *(optional)* | **NEW** | One-paragraph statement of what belongs in `docs/proposals/`, who reads them, and lifecycle (draft → accepted → implemented → archived) |

### 4.3 Removed from AGENTS.md after refactor

- The full Build Directory Setup recipe (→ `install_vortex.md`)
- The full Graphics extensions section (→ `extensions.md` or `microarchitecture.md`)
- The full per-driver test recipe (→ `testing.md`)
- The full `CONFIGS` override examples (→ `simulation.md`)

AGENTS.md keeps only the **one-line rule** + **link** for each.

---

## 5. Open Question: Public vs. Internal Workflow

`docs/contributing.md` describes the **public** GitHub fork-and-PR workflow. Several of the rules harvested from memory describe an **internal** workflow ("no PRs, direct commits to feature branches", "commit only at full feature completion"). These two are not contradictory — they apply at different stages — but **AGENTS.md must not surface the internal rules to public contributors as if they were the standard process**.

Proposed resolution (to be confirmed before drafting):

- `contributing.md` stays the public-facing doc.
- A new section in `contributing.md` titled **"Internal feature-branch workflow"** documents the direct-commit / substantial-commit / no-WIP rules and is explicitly scoped to maintainer-owned `feature_*` branches.
- AGENTS.md references both, and an agent operating on a `feature_*` branch is steered to the internal section.

Alternatives: (a) split into `contributing_public.md` + `contributing_internal.md`; (b) keep internal rules entirely inside AGENTS.md and out of the docs tree. **Preference: option above** (one `contributing.md` with two clearly-scoped sections) — it keeps the docs tree the single source of truth.

---

## 6. Phasing

The refactor is mechanical once the design is locked. Suggested phases, each landable independently:

**Phase 0 — Lock the design (this proposal)**
- Confirm the AGENTS.md skeleton in §4.1.
- Resolve the public/internal question in §5.
- Decide whether graphics extensions get their own doc or fold into `microarchitecture.md`.

**Phase 1 — Trunk doc updates**
- Update each existing doc per the table in §4.2 (no AGENTS.md changes yet).
- New `design_rules.md` lands here.
- Verifiable independently: existing AGENTS.md links keep working; new content is additive.

**Phase 2 — AGENTS.md rewrite**
- Replace AGENTS.md with the new skeleton.
- Move recipes out, link in.
- Add the "Read This First" foundation block.

**Phase 3 — Cross-branch propagation**
- Reconcile `dogfood/AGENTS.md` and `prototype1/AGENTS.md` (currently identical to each other, slightly drifted from `vortex_ci/AGENTS.md`).
- Decide whether these should be symlinks, generated copies, or independently maintained subset docs. **Preference: symlinks** to a single canonical AGENTS.md, since the rules apply uniformly.

**Phase 4 — Maintenance hooks** *(optional)*
- Add a brief CI lint check that fails if a `docs/*.md` file is modified without AGENTS.md's "last reviewed" date being touched, or vice versa. Mechanism TBD.

---

## 7. What's Explicitly Out of Scope

- Rewriting `microarchitecture.md`, `cache_subsystem.md`, etc. for technical accuracy. Those are content updates, separate from this organizational refactor.
- Generating an AI-specific document tree (e.g., `CLAUDE.md`, `GEMINI.md`, `.codex/`). The whole point is that one set of docs serves all agents. The empty `GEMINI.md` at the repo root and the per-tool agent files should be removed or pointed at AGENTS.md.
- Per-branch AGENTS variants. If a feature branch needs to record context, it goes in `docs/proposals/<feature>_proposal.md`, not in a branch-local AGENTS file.

---

## 8. Success Criteria

A new contributor (human or AI agent) opening the repo should be able to:

1. Read AGENTS.md cold and know **what conventions to follow** before writing any code — not just *how to run a test*.
2. Find the authoritative doc for any topic in **one link hop** from AGENTS.md.
3. Avoid the top ~15 well-known footguns (out-of-tree builds, `$HOME/tools`, re-`configure` after Makefile edits, ccache stale objects, `CONFIGS` matching, 120 s test cap, `RTL_PKGS` constraint, ...) without ever having been told individually.

If an agent's first non-trivial change to the repo triggers one of these footguns, that footgun is missing from AGENTS.md or its linked doc, and the fix is to **add it**, not to re-explain it.

---

## 9. Ask

Please review and flag any of the following before I start Phase 1:

- [ ] Skeleton in §4.1 — right shape, or do you want different top-level sections?
- [ ] Doc-update table in §4.2 — anything missing? Anything you'd rather not touch?
- [ ] Public/internal split in §5 — preferred resolution OK, or pick an alternative?
- [ ] Graphics extensions — new `extensions.md`, or fold into `microarchitecture.md`?
- [ ] Phase 3 cross-branch handling — symlinks, generated copies, or independent?
- [ ] Any additional rules from your own experience that should be on the "Read This First" list?
