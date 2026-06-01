// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_unpack — combinational walk of a 64 B cache line, extracting up to
// VX_CP_MAX_CMDS_PER_CL packed cmd_t records.
//
// Framing rules:
//   - Commands are byte-aligned but never cross a cache-line boundary.
//   - The runtime zero-pads to the end of the line if the next command
//     would overflow. A zero header (opcode=CMD_NOP=0, flags=0) terminates
//     the walk.
//
// Per-command on-wire layout:
//   [hdr (4B)] [arg0 (8B)] [arg1 (8B)] [arg2 (8B)] [profile_slot (8B)]
//   arg2 / profile_slot are present only for the opcodes that need them
//   (see cmd_size_bytes() in VX_cp_pkg.sv). Bytes are little-endian.
// ============================================================================

module VX_cp_unpack
  import VX_cp_pkg::*;
#(
  parameter int MAX_CMDS = VX_CP_MAX_CMDS_PER_CL_C
)(
  input  wire  [CL_BITS-1:0]                cl_data,
  output logic [$clog2(MAX_CMDS+1)-1:0]     cmd_count,
  output cmd_t                               cmds [MAX_CMDS]
);

  // Read byte `idx` straight out of the packed cache line via a dynamic
  // part-select. (Equivalent to an unpacked byte array, but a packed
  // dynamic part-select synthesizes cleanly under sv2v/yosys, whereas a
  // dynamically-indexed unpacked array does not.)
  function automatic logic [7:0] cl_byte(input int idx);
    return cl_data[idx*8 +: 8];
  endfunction

  // Extract a little-endian 64-bit value from offset `off` in the line.
  function automatic logic [63:0] read64(input int off);
    logic [63:0] v;
    v = '0;
    for (int i = 0; i < 8; ++i) begin
      if (off + i < CL_BYTES)
        v[i*8 +: 8] = cl_byte(off + i);
    end
    return v;
  endfunction

  // Extract the 4-byte header at offset `off`.
  function automatic cmd_header_t read_hdr(input int off);
    cmd_header_t h;
    h = '0;
    if (off + 0 < CL_BYTES) h.opcode   = cl_byte(off + 0);
    if (off + 1 < CL_BYTES) h.flags    = cl_byte(off + 1);
    if (off + 2 < CL_BYTES) h.reserved[7:0]  = cl_byte(off + 2);
    if (off + 3 < CL_BYTES) h.reserved[15:8] = cl_byte(off + 3);
    return h;
  endfunction

  // Walk the line, decode one command at a time until end-of-line or
  // a zero-header (padding) sentinel.
  always_comb begin
    // `automatic` because an always_comb evaluates fresh on every input
    // change; we don't want stale latched values across iterations.
    // Initialize up front so verilator's combinational-latch analysis
    // doesn't flag the conditional `sz = ...` inside the loop.
    automatic int                 offset   = 0;
    automatic cmd_header_t        hdr      = '0;
    automatic int unsigned        sz       = 0;
    automatic int unsigned        count    = 0;
    automatic cp_opcode_e         op       = CMD_NOP;
    automatic logic               profiled = 1'b0;

    // Default outputs.
    cmd_count = '0;
    for (int i = 0; i < MAX_CMDS; ++i) begin
      cmds[i] = '0;
    end
    for (int slot = 0; slot < MAX_CMDS; ++slot) begin
      // Stop if there isn't even room for a 4 B header in the line.
      if (offset + 4 > CL_BYTES) begin
        // exit loop
      end else begin
        hdr      = read_hdr(offset);
        op       = cp_opcode_e'(hdr.opcode);
        profiled = hdr.flags[F_PROFILE];

        // Zero header = padding to end of line; stop here.
        if (hdr.opcode == 8'h00 && hdr.flags == 8'h00) begin
          // exit loop
        end else begin
          sz = cmd_size_bytes(op, profiled);
          if (offset + int'(sz) > CL_BYTES) begin
            // Malformed line (a command would cross the CL boundary);
            // treat as end-of-line so the CPE doesn't dispatch garbage.
            // exit loop
          end else begin
            cmds[slot].hdr  = hdr;
            cmds[slot].arg0 = read64(offset + 4);
            cmds[slot].arg1 = read64(offset + 4 + 8);
            cmds[slot].arg2 = read64(offset + 4 + 16);
            cmds[slot].profile_slot = profiled
              ? read64(offset + int'(sz) - 8)
              : 64'd0;
            count = count + 1;
            offset = offset + int'(sz);
          end
        end
      end
    end

    cmd_count = ($clog2(MAX_CMDS+1))'(count);
  end

endmodule : VX_cp_unpack
