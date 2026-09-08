# Optional S2G line-pipeline

`VX_dxa_s2g_data_pipe` is selected with `-DVX_CFG_DXA_S2G_PIPELINED` and is
otherwise absent from the default worker datapath.  It keeps a small array of
line payload slots (`VX_CFG_DXA_S2G_PIPE_SLOTS`, default 4):

```text
addr_gen -> free payload slot -> LMEM gather -> complete payload slots -> GMEM store
              ^                 |                              |
              |                 +-- source-consumed completion +-- ready/valid
              +---- credit (ag_ready)
```

Unlike the legacy S2G FSM, a global-store request can be accepted while the
next line is being gathered.  The optional multi-read mode also keeps up to
`VX_CFG_DXA_S2G_READ_CREDITS` LMEM reads in flight (defaults to the slot count).
Each request carries its
payload slot and word index in the extension bits of the LMEM tag, so responses
may return out of order without a sideband reorder RAM.

The `VX_CFG_DXA_S2G_PIPE_MULTI_READ` switch adds
`slot_bits + word_bits` bits to `DXA_LMEM_TAG_W`. Both widths derive from the
configured slot count, cache-line size, and LMEM word size. The
encoding is:

```text
tag.value = {word_index, slot_index, core_id, dxa_engine_bit}
```

`VX_dxa_s2g_data_pipe` issues up to `READ_CREDITS` LMEM requests,
incrementing `words_issued[slot]` on each handshake.  A response decodes
`{slot,word}` and writes directly into that slot's payload; a small
`words_seen[slot]` bitmap detects malformed duplicate responses, which fail a
runtime assertion; this is not a retry/recovery protocol. `words_done[slot]` sets `complete` when it
reaches `word_count`.  The existing
`VX_dxa_core.sv`/`VX_socket.sv` tag adapters already derive their widths from
`DXA_LMEM_TAG_W` and `DXA_LMEM_OUT_TAG_W`, so the macro preserves the default
route and makes the extra width explicit for synthesis assertions.

For example, with 64-byte GMEM lines and 16-byte LMEM words, two accepted
address-generator tokens can be gathered as follows:

```text
cycle       0       1       2       3       4       5       6       7
request     S0/W0   S0/W1   S0/W2   S0/W3   --      S1/W0   S1/W1   --
response    --      --      --      --      S0/W2   S0/W0   S0/W3   S1/W1
capture                                    S0[2]   S0[0]   S0[3]   S1[1]
```

The existing route is `core_id` plus the DXA-engine bit;
the extension is `{word_index, slot_index}`. `read_pending_r` is the credit
counter. It reaches four in this example, then stops issuing until a response
decrements it. Completion is per slot, not per response order.
Thus a slot is emitted only after all of its words have been captured, and the
group tracker receives exactly one `SOURCE_CONSUMED` event for the architectural
S2G operation.

`SOURCE_CONSUMED` is generated once, after the final address-generator token
has arrived and every accepted line has captured all source words.  It is
independent of destination visibility.  OOB tokens consume no payload slot;
empty descriptors enqueue their source event at transfer start and retain it
until the completion sink accepts it. The architectural completion carries
only `{core_id, wid, group_id}`; worker payload tags are independent.

The payload geometry is derived from `DXA_LMEM_WORD_SIZE` and
`GMEM_LINE_SIZE`; byte offsets and byte-enable masks therefore handle either
word being smaller than a line or a word spanning a line boundary.  The
implementation does not add cache operations, destination/full waits, or
multicast behavior.

## Slot lifetime and backpressure

Completed lines may emit out of address-generation order. Consequently an
allocation searches for an actually invalid slot; a cyclic pointer combined
only with a free-count is insufficient. For example, if slot 0 has a delayed
read and slot 1 emits first, the next token must reuse slot 1, not slot 0.

The selected LMEM request and selected GMEM payload slot are held while their
respective ready signal is low. A newly ready lower-numbered slot cannot change
an already offered request. The worker is reusable only when the final address
token has arrived, **every payload slot has emitted**, and the source event has
been accepted. Neither emission of the `last` token nor an OOB final token
proves that earlier payload slots have drained. This is local worker drain,
not architectural destination completion or visibility.

For a 64 B cache line, 16 B LMEM word, 4 slots and 4 credits:

- At most 5 source words are needed for an unaligned line, so the word index is
  3 bits; slot index is 2 bits; LMEM tags grow by 5 bits.
- The outstanding-read counter is 3 bits. Slot counters use `ceil(log2(5+1))`
  bits, not an index width which could wrap when word counts are powers of two.
- Each slot has a 5-bit received-word bitmap. Holding both selected request
  slots costs two valid bits and two 2-bit indices.
- Payload storage remains `slots * max_source_words * LMEM_word_bytes` (320 B
  here). Larger LMEM words can substantially increase it; no area/Fmax claim is
  made without synthesis.

## Validation and limits

`hw/unittest/dxa_s2g` is a **Verilator RTL unit**, not the SimX model. It delays
every eleventh read, selects younger ready responses first, transfers 24 rows
and an unaligned 20-line tile through the smaller slot pool, stalls both memory
request interfaces, and overwrites SMEM immediately after SOURCE_CONSUMED.
Checks cover request stability, actual out-of-order returns, all destination
bytes/canaries, exact-once source events, OOB/zero-length transfers, and global
stores still pending when SOURCE_CONSUMED fires.

The strengthened test against the original pipe produces 168 failures,
including only 21 of 24 stores emitted and payload changes under backpressure.
Fixed-pipe validation includes non-power-of-two slot counts and LMEM words
smaller and larger than a cache line; exact commands/results are recorded in
`dxa_s2g_lmem_validation.md`.

**SimX timing parity and end-to-end kernel integration have not been updated
or validated for this datapath.** Earlier SimX kernel sweeps using the old
serial worker do not measure this optimization. No kernel speedup follows from
these unit tests. Full-chip/performance runs remain a separate required step
on `orcas2.cs.ucla.edu`, with L2 enabled.
