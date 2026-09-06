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
next line is being gathered.  The optional multi-read mode also keeps up to
`VX_CFG_DXA_S2G_PIPE_SLOTS` LMEM reads in flight.  Each request carries its
payload slot and word index in the extension bits of the LMEM tag, so responses
may return out of order without a sideband reorder RAM.

The reserved `VX_CFG_DXA_S2G_PIPE_MULTI_READ` switch adds
`slot_bits + 4` bits to `DXA_LMEM_TAG_W` (commit `369dfbae3`).  The intended
encoding is:

```text
tag.value = {word_index[3:0], slot_index, core_id, dxa_engine_bit}
```

`VX_dxa_s2g_data_pipe` issues up to `READ_CREDITS` LMEM requests,
incrementing `words_issued[slot]` on each handshake.  A response decodes
`{slot,word}` and writes directly into that slot's payload; a small
`words_seen[slot]` bitmap makes completion exact-once even if a malformed
duplicate response is presented.  `words_done[slot]` sets `complete` when it
reaches `word_count`.  The existing
`VX_dxa_core.sv`/`VX_socket.sv` tag adapters already derive their widths from
`DXA_LMEM_TAG_W` and `DXA_LMEM_OUT_TAG_W`, so the macro preserves the default
route and makes the extra width explicit for synthesis assertions.  The
The unit test drives variable latency and deliberately chooses the newest
ready response first.  The four-line case reaches four simultaneous LMEM
reads and passes the same memory checks as the default one-credit path.

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
