---
name: code-review
description: Review a Vortex diff (a commit, a branch range or a pull request) against the repository's own rules, classify its change impact (R0-R3) and report findings by severity (F0-F3). Use before landing any change, when asked to review a PR or a commit range, and as the independent reviewer an R3 change requires. Static review; it never edits, commits or pushes.
---

# Reviewing a Vortex change

The policy is [docs/review/code-review-policy.md](../../../docs/review/code-review-policy.md).
This skill is how to apply it.

## 1. Load the rules

Read these in full before reading the diff. They are not auto-loaded.

- `AGENTS.md`
- `docs/coding_guidelines_verilog.md`, `docs/coding_guidelines_cpp.md`
- `docs/bug_fixes.md`
- `docs/review/code-review-policy.md`

For RTL hunks also follow the `rtl-review` skill; for `sim/simx/` hunks, the
`simx-review` skill.

## 2. Classify

```bash
python3 ci/change_impact.py classify --base <base> --head <head>
python3 ci/change_impact.py explain  --base <base> --head <head>   # per file, with rules
```

The class decides the depth: R0 needs no review, R1 a read of the change, R2
and R3 every changed hunk. Each contract finding and each evidence item the
classifier prints is something the review must confirm or refute. The
classifier is advisory: if the diff carries a contract change it missed, say so
and treat the change as R3.

## 3. Review the diff

Review the diff, not the files around it. Comment on unchanged code only where
the diff makes it wrong. For each hunk ask:

- Does it do what the commit message says, on every configuration the toml and
  `ci/regression.sh` allow, at XLEN 32 and 64?
- What else consumes what it changed (an interface, a struct, a CSR, a DCR, a
  header, a testcase, a baseline)? Did those consumers change with it?
- Can it move cycles? Then where is the SimX side, and which `model_parity`
  case shows it?
- Does it fix a root cause, or move a symptom (`docs/bug_fixes.md` §2)?

For a range or a branch, also review the final state of every file that
several commits touched: the end state matters more than each step.

A large or cross-subsystem diff goes to a fresh reviewer that did not write
it, with the diff and the rule documents; an R3 change always does.

## 4. Hard rules: always F2 or F3

| rule | why |
|---|---|
| golden data (`ci/baselines/**`, reference images) changed without the update command in the commit message, or re-recorded by an agent or CI | a hand edit absorbs a regression; only a human re-records golden data |
| a test silenced: `known_issue:`, wider `tolerance:`/threshold, smaller `args:`, a removed case, to make a failure go away | fix the bug instead |
| an existing test case changed or removed, a golden value moved toward a looser bar (more cycles, lower Fmax, more area), or a gate threshold widened, without the stated, measured reason and the maintainer's approval | these change what CI proves (R3: `test-coverage`, `golden-values`, `gate-thresholds`) |
| a bug papered over (fallback, clamp, delay, retry, disabled feature) without a `PATCH:`/`WORKAROUND:` label and a follow-up | `docs/bug_fixes.md` §3 |
| `sw/kernel` or `sw/runtime` references `hw/` or `sim/`, or the reverse | layer isolation; shared ABI goes in `sw/common` |
| a Verilog interface added to `RTL_PKGS` | `RTL_PKGS` is `VX_*_pkg.sv` only |
| a comment names a task, PR, issue, phase or "the fix", or names the other layer's internals | comments explain why, and rot otherwise |
| RTL timing changed without SimX, or the reverse, with no parity evidence | SimX is the RTL's timing model |
| a toolchain pin moved before the prebuilt tag exists | CI downloads binaries at the pinned tag |
| a switch that is parsed and ignored | worse than no switch |
| a workflow exposes a self-hosted runner or secrets to a fork | trust boundary |

Do not report style preferences, naming taste or anything the documents do not
require.

## 5. Report

One entry per finding:

```
F2 high static-proof  <sha> path/to/file.sv:123
What: <the defect, one sentence>
Scenario: <the configuration and input that make it fail, and what is observed>
Fix: <the smallest correct change>
```

- Severity F0-F3 and confidence as defined in the policy. Evidence type is
  `static-proof`, `reasoned-risk` or `missing-test`.
- Verify every finding against the code at the head of the range before
  reporting it. If it no longer holds there, drop it or say it was fixed later.
- State what you did not review and what evidence you could not see. A test you
  did not see run is `not run`, never a pass.
- End with the class, the obligations still open, and a verdict: land, land
  after the listed fixes, or do not land.

## 6. Never

- Edit, commit, push, or run `checkout`/`reset`/`stash`/`clean` while reviewing:
  the trees are shared.
- Run heavy builds (rtlsim, synthesis) to settle a review question without being
  asked; say what would settle it instead.
- Claim a result you did not see.
