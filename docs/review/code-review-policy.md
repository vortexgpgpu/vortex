# Code Review Policy

Normative. The rules that classify a change live in
[change-impact.yaml](change-impact.yaml) and are applied by
[ci/change_impact.py](../../ci/change_impact.py). The coding and process rules
a review checks against are [AGENTS.md](../../AGENTS.md),
[coding_guidelines_verilog.md](../coding_guidelines_verilog.md),
[coding_guidelines_cpp.md](../coding_guidelines_cpp.md) and
[bug_fixes.md](../bug_fixes.md). This policy adds no coding rules of its own.

## 1. Two scales

**Change impact (R)** says how much care a change needs before it lands. It is
a property of the diff, computed before review.

| class | meaning | source |
|---|---|---|
| R0 | no effect on what is built, run or enforced | paths; comment- or whitespace-only diffs |
| R1 | local effect its own tests cover | paths |
| R2 | shared RTL, the SimX model, the software stack, the build, CI enforcement, golden data | paths (the ceiling for a path rule), evidence checks |
| R3 | a detected externally visible change, or a change to what CI enforces | detectors only |

**Every change with external visibility is R3**, whether it adds, changes or
removes. R3 never comes from a path; it comes from a detector that finds, in
the diff content:

- an RTL interface signal, modport or parameter (`hw/rtl/interfaces/`);
- a `VX_gpu_pkg` typedef, or an `INST_*` ISA encoding in `VX_gpu_pkg.sv` or
  `VX_define.vh`;
- an instruction the decoder starts or stops accepting (`VX_decode.sv`,
  `sim/simx/decode.cpp`);
- a `VX_types.toml` value (CSR, DCR, opcode, memory map);
- a `VX_CFG_*` knob added, removed or renamed, or its default or derivation
  moved (`VX_config.toml`, `configure`, `ci/gen_config.py`);
- an installed or shared ABI header (`sw/runtime/include`, `sw/kernel/include`,
  the `sw/common` headers `Makefile.in` ships, the `sw/common` ABI headers):
  a symbol added or removed, a macro value, a struct, union or enum layout, a
  prototype, or an intrinsic's definition (`vx_intrinsics.h`);
- an install rule or list, an exported toolchain path, or an ABI/ISA flag
  (`-march`, `-mabi`, target features) in a non-test Makefile;
- a toolchain pin in `VERSION`;
- a new workflow trust exposure;
- an existing test case changed or removed in `ci/testcases/` (its `args:`,
  `configs:`, drivers, XLENs, tier, tolerance, `known_issue:` or any other
  key), or a file's defaults changed; a new case is not a finding;
- golden data changed or removed: a value in `ci/baselines/`, a reference
  image or the Beacon calibration; new entries are not a finding;
- a gate tolerance, threshold or margin moved in the gate code
  (`ci/perf_baseline.py`, `ci/*_gate.py`, `ci/testcase.py`, ...).

A diff that only touches comments or whitespace is never R3.
`change_impact.py replay` measures the R3 rate; there is no rate target.

**Finding severity (F)** says how bad a review finding is. It is a property of
the code, not of the process that produced it.

| severity | meaning | before merge |
|---|---|---|
| F3 | breaks a supported configuration, breaks a contract without the other side, opens a trust boundary, or lets a gate pass a regression | fix |
| F2 | a rule violation or likely bug with a concrete failure scenario on a reachable configuration, or a missing obligation of the change's class | fix, or a waiver the maintainer grants (§6) |
| F1 | a latent risk with a narrow trigger, or a document that now disagrees with the code | follow-up |
| F0 | an observation | none |

Every finding states its **confidence** (high, medium, low) and its **evidence
type**: `static-proof` (the code shows it), `reasoned-risk` (a failure path
argued from the code), or `missing-test` (nothing exercises it). A finding
without a concrete failure scenario is not a finding.

## 2. What each class requires

Vortex has one maintainer. No rule below asks for a second human.

| class | review | evidence | approval |
|---|---|---|---|
| R0 | none | CI | none |
| R1 | author | the change's own tests | none |
| R2 | a review of every changed hunk against the rules, with the `rtl-review` or `simx-review` skill for the areas touched | named in the PR: what ran, what did not and why; Beacon (`logic_gate`) for RTL that can move timing; a `model_parity` case for a change that moves cycles | maintainer |
| R3 | R2, plus a review agent that did not write the change | R2, plus the obligation of the contract (§3) | maintainer, recorded in the PR |

An R3 change from an outside contributor needs the maintainer's approval;
the review agent's report is advisory input to it.

**A test that did not run is not a pass.** Evidence reports `run`, `not run`
or `failed`, and `not run` never counts toward an obligation.

## 3. Contract obligations

| detector | the change must also show |
|---|---|
| `rtl-interface`, `gpu-pkg-types` | every instantiation updated in the same change; rtlsim or xrt elaborates; Beacon clean on the units that use it; the SimX mirror of the struct, if any, matches |
| `isa-localparams`, `isa-decoder`, `isa-encoding` | the RTL decoder, SimX, runtime, kernel library and LLVM agree; an ISA test covers the instruction; a toolchain change follows the landing order (AGENTS.md §2) |
| `config-knobs` | the profiles, the CI catalog and the docs that name the knob are updated; a moved default names the parity and perf cases that show its effect, and the perf baselines are re-recorded by a human, never by an agent or CI |
| `sw-common-abi`, `installed-api` | the host writer and the device or model reader change together; the downstream consumers (PoCL, Mesa, chipStar, tests) build, or the prebuilts are refreshed first |
| `build-surface` | the installed tree and the toolchain it names are checked from a clean `make install` |
| `toolchain-pins` | the prebuilt tag exists with matching binaries before the change lands |
| `workflow-trust` | the exposure is intended and stated; no self-hosted runner is reachable from a fork |
| `test-coverage` | each changed or removed case states why, with the measurement behind it; a coverage cut or a silenced failure carries the root-cause plan of [bug_fixes.md](../bug_fixes.md) §3 |
| `golden-values` | the update command that regenerated it, run by a human; the per-entry delta, with every value that moved toward a looser bar (more cycles, lower Fmax, more area) attributed to a commit and fixed or justified |
| `gate-thresholds` | the measured reason the bound must move; a widened bound is never the fix for a red gate |

## 4. Evidence checks

These report; they never raise a change above R2.

- **SimX/RTL lockstep.** RTL logic changed without SimX, or the reverse. The
  PR names the `model_parity` case that shows no cycle moved, or carries the
  matching model change.
- **Golden-data provenance.** A file under `ci/baselines/` or a reference image
  changed (also R3 through `golden-values` when an existing value moved). The
  commit message names the update command that regenerated it.
  Golden data is never hand-edited, and only a human re-records it, never an
  agent or CI.
- **Silenced tests.** A `known_issue:`, a wider `tolerance:`/threshold, smaller
  `args:` or a removed case (also R3 through `test-coverage`). Each needs the root-cause plan of
  [bug_fixes.md](../bug_fixes.md) §3, or is reverted.
- **Toolchain pin.** `TOOLCHAIN_REV` or another pin in `VERSION` moved (also R3
  through `toolchain-pins`). The prebuilt tag exists before the Vortex change
  lands.
- **configure-copied files.** A Makefile, `*.mk` or `*.in` changed; tests ran
  from a re-configured build tree.

## 5. Trust boundaries

- CI classifies with the **base revision's** copy of the classifier and this
  policy, so a change to either file takes effect for the changes after it.
- The CI summary is **untrusted** for a change that touches `.github/**` or
  `.gitattributes`: such a change can alter the job that runs the classifier,
  or how git presents the diff to it. The reviewer reruns the base copy of the
  classifier locally instead:
  ```bash
  B=$(git merge-base origin/master HEAD); D=$(mktemp -d)
  git show $B:ci/change_impact.py > $D/change_impact.py
  git show $B:docs/review/change-impact.yaml > $D/change-impact.yaml
  python3 -I $D/change_impact.py --repo . --policy $D/change-impact.yaml \
    explain --base $B --head HEAD
  ```
- The classifier only reads the repository. It never posts, comments or
  approves; its CI job writes the result to the job summary, never fails, and
  no other job depends on it.
- A pull request always has a merge base. A push without a usable base (new
  branch, force push) is classified R3.

## 6. Waivers

Only the maintainer grants a waiver; neither the author nor an agent can
record its own. A waiver names the finding, why it is acceptable now, and the
follow-up that removes it, in the PR description. A waiver that expires after merge becomes a
tracked issue the same day; it is not renewed silently.

## 7. Changing the policy

Bump `policy_version` in `change-impact.yaml` on any rule change, run
`python3 ci/change_impact.py lint-policy` and
`python3 -m pytest ci/test_change_impact.py`, and report the R distribution of
`change_impact.py replay --range origin/master --max-count 300` in the commit
message. Test selection stays in `ci/testcases/*.yaml`; this policy never maps
paths to tests.
