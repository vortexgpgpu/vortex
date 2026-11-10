## Summary

<!-- What changes and why. The diff shows how. -->

## Impact and evidence

See [docs/review/code-review-policy.md](../docs/review/code-review-policy.md).

**Impact class:** R? <!-- from the CI change-impact job's summary or
`python3 ci/change_impact.py classify --base origin/master --head HEAD`; if you
disagree with it, give the class you claim and why -->

**Contracts touched:** none <!-- interface, VX_gpu_pkg struct, VX_types.toml
value, sw/common ABI struct, installed header, workflow; and how the other side
changed with it -->

| evidence | status | detail |
|---|---|---|
| functional tests (simx / rtlsim / xrt) | run / not run | |
| `model_parity` | run / not run / n/a | case names; cycle gap |
| `perf_gate` | run / not run / n/a | baseline regenerated with `--update-baselines`? |
| Beacon `logic_gate` | run / not run / n/a | DUTs; deeper units and why |
| synthesis (`fpga_gate` / `asic_gate`) | run / not run / n/a | |
| XLEN 64 build | run / not run | |

A `not run` is not a pass. Say why it was not run.

- [ ] No test silenced (`known_issue:`, wider tolerance, smaller args, removed case) without a root-cause plan.
- [ ] Golden data changed only through its update command, named in the commit message.
- [ ] Cycle-moving RTL changes carry the matching SimX change, or the parity case that shows none moved.
- [ ] A toolchain pin moves only after the prebuilt tag exists.
