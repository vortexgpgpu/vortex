# Vortex feature worktree index

This checkout intentionally keeps architectural lines isolated.  Do not move,
clean, or merge a dirty worktree merely to make the directory look uniform.

| path | branch / role | review boundary |
|---|---|---|
| `repos/vortex-master` | `master` baseline | upstream checkout |
| `repos/vortex-master/worktrees/cfx-s2g-read-clean` | `cfx/s2g-read-clean` | S2G source-consumed groups and `wait_group_read<N>`; this task |
| `repos/vortex-master/worktrees/cfx-cache-visibility` | `cfx/cache-visibility` | cache/full-wait experiments; intentionally excluded here |
| `repos/vortex-master/worktrees/cfx-mcast-fabric` | `cfx/mcast-fabric` | multicast/CTA-cluster fabric line |
| `repos/vortex-master/worktrees/cfx-s2g-engine` | `cfx/s2g-engine` | earlier S2G engine experiments |
| `repos/vortex-master/worktrees/cfx-mcast-arch-explore` | `explore/simx-mcast-architectures-20260828` | simulation architecture exploration |
| `repos/vortex-master/worktrees/simx-*` | experiment branches | keep separate from reviewable feature branches |
| `repos/vortex-master/worktrees/simx-hier-mcast-explore` | `exp/simx-hier-mcast` | hierarchical multicast experiments |
| `repos/vortex-master/worktrees/simx-mcast-factorial` | `explore/simx-mcast-factorial-20260828` | multicast parameter sweep |
| `repos/vortex-master/worktrees/vortex-dxa-ctrl` | detached DXA control snapshot | historical/reference checkout |
| `repos/vortex-master/worktrees/vortex-dxa-store-proto-20260812` | `smem-amo-socket-dxa` | earlier socket-DXA/SMEM prototype |
| `repos/vortex-master/worktrees/vx-dxa-group-s2g` | `feat/dxa-group-s2g` | earlier S2G/group prototype; do not merge blindly |
| `kitsune/worktrees/vortex-gbar-response-fix` | `fix/gbar-response-phase` | global-barrier response-phase PR line |
| `/tmp/vortex-cache-baseline-fed93` | detached cache baseline | temporary comparison only |

There are also two intentionally separate repositories under `kitsune/worktrees`:
the FireSim checkout (`firesim-vortex-4c-s4-nt16-nw8-wgmma`) and the paper
exploration checkout (`paper-eval-0728`). They are not Vortex worktrees and
should not be included in a Vortex feature commit.

Generated build products are currently left in place because the working
directory is shared with other experiments.  For a review/commit, stage only
source, tests, and design documents; leave `obj_dir/`, simulator binaries,
config stamps, and logs out of the patch.  The canonical review order is:

```text
gbar response-phase branch  -> independent PR
S2G read-group branch      -> independent PR/review
cache/full-wait branch      -> separate design and PR
multicast/cluster branches  -> separate architecture experiments
```
