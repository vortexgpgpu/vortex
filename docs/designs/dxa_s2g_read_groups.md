# S2G read groups

See [the current counter-ring design](dxa_s2g_simple_groups.md) for the state,
tokens, ordering, source-consumption contract, and scheduler integration.

The public programming path is grouped S2G issue, `vx_dxa_commit_group()`, and
`vx_dxa_wait_group_read<N>()`. Ownership is per issuing Vortex warp. Raw
ready/free barriers propagate source ownership to other warps in the CTA.

Representative workloads:

- [TMA + WGMMA pipelined epilogue](../../tests/regression/dxa_tma_wgmma_s2g_pipeline/README.md)
- [Attention](../../tests/regression/dxa_tma_attention_s2g/README.md)
- [WGMMA epilogue](../../tests/regression/dxa_s2g_wgmma_epilogue/README.md)

Directed tests remain necessary: a realistic kernel does not guarantee that
empty groups, same-cycle races, full rings, early completions, and invalid
completion targets occur deterministically.

The separate cache-visibility/full-wait worktree is not part of this branch.
