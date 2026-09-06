# Optional S2G line-pipeline

`VX_dxa_s2g_data_pipe` is selected with `-DVX_CFG_DXA_S2G_PIPELINED` and is
otherwise absent from the default worker datapath.  It keeps a small array of
line payload slots (`VX_CFG_DXA_S2G_PIPE_SLOTS`, default 4):

```text
addr_gen -> slot FIFO -> LMEM gather -> complete payload slots -> GMEM store
              ^                 |                              |
              |                 +-- source-consumed completion +-- ready/valid
              +---- credit (ag_ready)
```

Unlike the legacy S2G FSM, a global-store request can be accepted while the
next line is being gathered.  The source side currently has one outstanding
LMEM read because the existing DXA LMEM tag is only `{uuid, core, engine}`;
adding read credits requires extending that tag route or using a per-port
response reorder table.  The line FIFO still removes store-side head-of-line
blocking and provides the integration point for that future credit extension.

`SOURCE_CONSUMED` is generated once, after the final address-generator token
has arrived and every accepted line has captured all source words.  It is
independent of destination visibility.  OOB tokens consume no payload slot;
empty descriptors complete at transfer start.  A generation/operation token
continues to be supplied by the existing group tracker.

The payload geometry is derived from `DXA_LMEM_WORD_SIZE` and
`GMEM_LINE_SIZE`; byte offsets and byte-enable masks therefore handle either
word being smaller than a line or a word spanning a line boundary.  The
implementation does not add cache operations, destination/full waits, or
multicast behavior.
