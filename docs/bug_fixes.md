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

## 6. Related rules

- AGENTS.md §3 — short-form bug-fix rules
- AGENTS.md §6 — comment rules (no `// fix for #123`-style breadcrumbs in code)
- [CONTRIBUTING.md](../CONTRIBUTING.md) — public PR flow, including how reviewers gate on regression coverage
