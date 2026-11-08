# Bug-Fix Discipline

A bug is a symptom. The *fix* is whatever change makes the underlying cause go away. Anything else is a workaround — useful sometimes, but it must be labeled as such and tracked separately. This doc captures the rules Vortex follows when diagnosing and fixing bugs.

## 1. Root cause before patch

- **Diagnose first.** Before changing any code, understand *why* the bug happens. "The test passes now" is not a diagnosis.
- **Reproduce reliably.** A bug you can't reproduce is a bug you can't verify a fix for. Get to a deterministic repro before touching the code — even if that means crafting a smaller test, lowering simulator parallelism, or adding instrumentation first.
- **Fix the cause, not the symptom.** If a NaN propagates through a pipeline, the fix is in whatever produced the NaN — not in a downstream clamp. If a stall hangs the issue stage, the fix is in the unit that's not asserting `ready` — not in a watchdog that masks the hang.

## 2. What "papering over" looks like — and why we don't

These are the disguises a not-really-a-fix tends to wear. Each one is forbidden unless explicitly labeled as a temporary patch (see §3).

| Anti-pattern | Why it's bad |
|---|---|
| Suppressing or `-Wno-`'ing a warning | Warnings exist because the compiler/tool found something real. Silencing them hides future regressions in the same area. |
| Adding a fallback path that "handles the case" | Fallbacks accumulate. Each one becomes a permanent maintenance burden and a place where future bugs can hide. |
| Catching and ignoring an exception or assertion | Now the failure is silent. The next time the underlying condition occurs, no one will notice. |
| Hard-coding a value to "make the test pass" | The test passing was a goal, not a result. The code now lies about what it computes. |
| Bumping a timeout / retry count | If timing is *the* bug, this is a fix. If timing is *a symptom*, this just makes the bug intermittent. |
| Skipping the failing test | Acceptable only as a labeled patch (§3) with a tracked re-enable. Otherwise, the test bitrots into "always skipped." |
| Pinning to an older dependency version | Same as above — labeled patch, tracked unpinning, or it becomes permanent debt. |
| Adding `try ... catch (...) {}` around the problem area | The exception is now invisible. Future failures will not page anyone. |
| Reverting a clean commit because it "broke" something | Sometimes warranted, but only after understanding *why* it broke. Otherwise the same change has to be re-done later. |

## 3. When a patch is genuinely unavoidable

There are real situations where the proper fix can't land right now:

- The root cause is in an upstream dependency we don't control on this timeline.
- The proper fix touches a refactor that's larger than the current change window.
- A release deadline blocks the deeper fix and the patch unblocks others.

In those cases, a patch is acceptable, with **both** of these:

1. **Label it explicitly in the commit message.** Use a clear prefix — `PATCH:` or `WORKAROUND:` — and write *what* the workaround is, *why* the proper fix isn't being done now, and *what the proper fix would be*.
2. **File a follow-up.** A `TODO` in code is not a follow-up. A tracked issue / proposal / ticket is. Reference it in the commit message.

A patch without those two things is just a hidden bug.

### Patch-comment template

When a workaround is in code and needs an inline note, use this shape:

```cpp
// WORKAROUND: <one-line description of what's being worked around>
// Proper fix: <pointer to the issue / proposal / location of the real fix>
```

Keep it terse. The deep explanation lives in the linked issue, not in the comment.

## 4. After the fix

- **Add a regression test** that fails before the fix and passes after. If the bug was in RTL, add an rtlsim or xrt path that exercises it. If it was in the runtime, add a unit or smoke test.
- **Don't leave debug scaffolding behind.** Remove the prints, the extra logging, the temporary asserts — unless they're genuinely useful as long-term diagnostics, in which case gate them behind `--debug=` levels.
- **Audit nearby code.** A bug rarely lives alone. If you found a missing `ready` assertion in one unit, check siblings. If a kernel had a bad bounds check, check kernels that use the same template.

## 5. Anti-patterns specific to Vortex

A few footguns we've hit more than once:

- **"Fixing" a SimX-vs-RTL divergence by editing SimX.** SimX is the *reference*. If RTL diverges, the bug is almost always in RTL. Verify which side is wrong before touching the model.
- **Adding a `wait` or pipeline bubble to make a hang go away.** Stalls are flow-control bugs; they need the missing `ready`/`valid` handshake fixed, not a delay slot.
- **Disabling cache or memory features to dodge a hazard.** If a bug only repros with L2 enabled, the bug is in L2 — not a reason to ship with L2 off.
- **Changing default `CONFIGS` to mask a regression.** Defaults are a contract. If the regression test fails at the documented defaults, that's the regression — don't move the defaults.
- **Patching `ccache` symptoms.** `fmt::v8` link errors and similar stale-object symptoms are not bugs in the code — retry with `CCACHE_DISABLE=1` before changing anything.

## 6. Performance and area regressions

A gated metric that got worse is a bug report against the commit that moved it. The metrics are cycles and instruction counts (`perf_gate`), area (LUT, FF, BRAM, DSP), timing (Fmax, WNS, Beacon register depth), and power where a gate measures it. Find that commit first; the loss is recovered inside that commit's own change, or it is justified on that commit's terms. Improving another part of the system until the number returns is not a fix: it hides the regression, usually adds new ones, and leaves a baseline that mixes the regression with unrelated gains, so the next bisect cannot separate them.

1. **Separate the toolchain from the design.** For cycles, build the same RTL with the old and the new compiler. For area and timing, run the same RTL through the old and the new synthesis flow, tool version and constraints. A toolchain-only delta is fixed in the toolchain, never in RTL.
2. **Check that the case is stable.** Perturb it minimally: one cycle of latency for perf (for example `L1_LINK_LATENCY` ±1), a placement seed or directive for synthesis. If the metric swings past the gate tolerance, the case is bistable: attribute nothing to it and tune nothing on it. Fix the case or the pathology behind it, as its own change.
3. **Bisect every slower case to one commit** and explain the mechanism in that commit's terms. The result is a table: case, commit, mechanism, class. No fix is written before the table is complete, and the table is reviewed before any fix lands or any baseline is re-recorded.
4. **Classify each case:**

   | class | meaning | outcome |
   |---|---|---|
   | bug | the commit does something it did not intend, or pays more than its goal requires | fixed inside that commit's change |
   | trade-off | the commit pays this cost for a gain measured elsewhere (Fmax, area, DRAM traffic, cycles on other cases) | justified with both numbers; the owner decides |
   | measurement correction | the model now counts what the silicon always had (cycles in rtlsim or SimX, area in an estimator) | justified and recorded; the old baseline was wrong |
   | toolchain | the compiler or synthesis flow changed its output | classified again, inside the toolchain, with the same four classes |

5. **Prove each justification; a plausible story is not enough.** "Trade-off", "measurement correction" and "correctness fix" are claims, and each one must pass all of these:
   - **Goal.** State what the commit set out to do, in its own terms.
   - **Prediction.** From the mechanism, predict the direction and rough size of the cost before looking at the gate. A cost that the mechanism does not predict is not explained yet.
   - **Inherent, not avoidable.** Show that the commit's goal cannot be met at a lower cost: name the cheapest correct alternative and show that it is worse or not available. A correctness fix is justified only for the cost that correctness itself requires; extra cost the fix leaves behind (an avoidable reload, an extra pipeline stage, a register the design did not need) is a bug in the fix.
   - **Ownership.** If the cost appears only through another, pre-existing bottleneck, prove it: remove that bottleneck experimentally and show that the commit's cost disappears. The cost then belongs to the bottleneck, which gets its own proposal (step 8).
   - **Measurement correction.** Show the silicon already had the cost: the commit's diff leaves the synthesized logic unchanged, and the model now matches the hardware cycle for cycle on a bench.
   - **Evidence by experiment.** Attribute by building or simulating with only the suspect change present or absent, never by reading code alone.
6. **A fix touches only the culprit:** its code, its parameter or default, or a revert of it.
7. **Stop rule.** If a fix fails parity or worsens other cases or metrics, and making it pass needs a change in another subsystem, stop and reclassify that culprit. Do not follow the fix into the rest of the design.
8. **Problems found on the way are separate changes.** A defect that already exists on the base (for example an arbiter that starves a unit) gets its own proposal, commit and measurement against the refreshed baseline. It is never folded into the regression fix or the re-record.
9. **Keep the window small.** Re-record baselines when a feature lands, and treat a gate that is masked or stale as a CI bug. Attribution over hundreds of commits is the failure this rule exists to prevent.

## 7. Related rules

- AGENTS.md §3 — short-form bug-fix rules
- AGENTS.md §6 — comment rules (no `// fix for #123`-style breadcrumbs in code)
- [CONTRIBUTING.md](../CONTRIBUTING.md) — public PR flow, including how reviewers gate on regression coverage
