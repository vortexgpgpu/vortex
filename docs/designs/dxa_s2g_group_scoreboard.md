# S2G source-group scoreboard

NVIDIA's `DEPBAR.LE SB0,N` is best treated as an architectural wait on an
implicit dependency-scoreboard slot. `SB0` is not an ordinary integer/XREG
register: software cannot load it, spill it, or use it as a WGMMA accumulator.
The public ISA specifies the ordered group operation, while the physical
counter and wakeup implementation are undocumented.

The Vortex implementation keeps the same useful separation but makes the
state explicit. Each warp owns one open accumulator and a FIFO of sealed
groups:

```text
warp w
  grouped S2G issue ──> open[w] {seq, pending source operations}
  commit_group      ──> ring[w][tail] {sealed, pending, generation}
  source complete   ──> token {wid, slot, generation, seq}
  wait_read<N>      ──> stall w until ordered head retirement leaves N groups
```

For example, the sequence

```text
store A; store B; commit       // G0
store C; commit                 // G1
wait_group_read<1>()
```

must wait for G0 even if B and C complete before A. A single aggregate
counter would incorrectly release the warp; the ring checks the head in commit
order. `N=1` retains G1 and permits the stage owned by G0 to be reused.

The existing Vortex XREG scoreboard tracks register read/write hazards and
functional-unit reservations. It is not a substitute for this ring: a source
completion may arrive while the warp is descheduled, and completion order can
differ from commit order. The scheduler may nevertheless reuse its normal
park/unlock path: `wait_read` parks only the issuing warp, while the group ring
is a small hardware RAM indexed by `{wid, slot}`. Thus this is not a software
stack in registers or shared memory, and it does not require a replicated
tracker per thread.

The optional line-pipelined S2G worker uses the same group token and emits one
`SOURCE_CONSUMED` completion after all source words have entered a stable line
payload slot. The current LMEM ABI still carries `{uuid, engine, core}` and
therefore permits one outstanding source read. `VX_CFG_DXA_S2G_PIPE_MULTI_READ`
only reserves extra `{slot, word}` tag bits; it is deliberately disabled until
the response reorder table is implemented. The default path is therefore
correct rather than silently assuming in-order LMEM responses.

Non-goals of this design are destination completion, cache visibility, cache
invalidation, and full wait. Those need separate completion domains and must
not be inferred from `SOURCE_CONSUMED`.
