# S2G source-group control

The current implementation is documented in
[S2G source groups: per-warp counters](dxa_s2g_simple_groups.md).

Each warp owns a fixed ring of group pending counters. The open group reserves
the tail row on first issue; commit closes that row. Worker source completions
carry only `{core_id, wid, group_id}`. There is no separate operation-context
table or software stack. A read wait uses scheduler park/unlock without
consuming a transaction-barrier slot.

This mechanism establishes source-buffer reuse safety, not destination
completion or cache visibility. It makes no claim about NVIDIA's undocumented
physical implementation.
