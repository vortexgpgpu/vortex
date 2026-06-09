// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_unpack — single-command decoder.
//
// Decodes exactly ONE packed cmd_t record located at byte `offset` within a
// 64 B cache line. The fetch FSM (VX_cp_fetch) walks the line by registering
// `offset` and advancing it by `cmd_size` after each command is emitted.
// One command is decoded per cycle; the offset accumulation is a single
// registered add (~4 logic levels).
//
// Per-command on-wire layout (little-endian, byte-aligned, never crosses a CL):
//   [hdr (4B)] [arg0 (8B)] [arg1 (8B)] [arg2 (8B)] [profile_slot (8B)]
//   arg2 / profile_slot present only for the opcodes that need them
//   (see cmd_size_bytes() in VX_cp_pkg.sv).
//
// has_cmd is deasserted (end-of-line) when:
//   - there isn't room for a 4 B header (offset + 4 > CL_BYTES), or
//   - the header is a zero (opcode==0 && flags==0) padding sentinel, or
//   - the command would overrun the cache line (offset + size > CL_BYTES).
// ============================================================================

module VX_cp_unpack
  import VX_cp_pkg::*;
(
  input  wire  [CL_BITS-1:0]   cl_data,
  input  wire  [OFF_W-1:0]     offset,      // byte offset to decode at
  output logic                 has_cmd,     // 1 = a valid command at offset
  output cmd_t                  cmd,
  output logic [OFF_W-1:0]      cmd_size     // bytes consumed (for offset advance)
);

  // Offset/size domain: 0 .. CL_BYTES (need to represent CL_BYTES itself).
  localparam int OFF_W = $clog2(CL_BYTES + 1);

  // Read byte `idx` straight out of the packed cache line via a dynamic
  // part-select (synthesizes cleanly under sv2v/yosys, unlike a dynamically
  // indexed unpacked array).
  function automatic logic [7:0] cl_byte(input int idx);
    return cl_data[idx*8 +: 8];
  endfunction

  // Little-endian 64-bit value from offset `off`.
  function automatic logic [63:0] read64(input int off);
    logic [63:0] v;
    v = '0;
    for (int i = 0; i < 8; ++i) begin
      if (off + i < CL_BYTES)
        v[i*8 +: 8] = cl_byte(off + i);
    end
    return v;
  endfunction

  // 4-byte header at offset `off`.
  function automatic cmd_header_t read_hdr(input int off);
    cmd_header_t h;
    h = '0;
    if (off + 0 < CL_BYTES) h.opcode         = cl_byte(off + 0);
    if (off + 1 < CL_BYTES) h.flags          = cl_byte(off + 1);
    if (off + 2 < CL_BYTES) h.reserved[7:0]  = cl_byte(off + 2);
    if (off + 3 < CL_BYTES) h.reserved[15:8] = cl_byte(off + 3);
    return h;
  endfunction

  always_comb begin
    automatic int            off = int'(offset);
    automatic cmd_header_t   hdr;
    automatic cp_opcode_e    op;
    automatic logic          profiled;
    automatic int unsigned   sz;

    cmd      = '0;
    has_cmd  = 1'b0;
    cmd_size = '0;

    // Need room for at least a 4 B header.
    if (off + 4 <= CL_BYTES) begin
      hdr      = read_hdr(off);
      op       = cp_opcode_e'(hdr.opcode);
      profiled = hdr.flags[F_PROFILE];

      // Zero header = padding to end of line → end-of-line.
      if (!(hdr.opcode == 8'h00 && hdr.flags == 8'h00)) begin
        sz = cmd_size_bytes(op, profiled);
        // Reject a command that would cross the CL boundary (malformed).
        if (off + int'(sz) <= CL_BYTES) begin
          cmd.hdr          = hdr;
          cmd.arg0         = read64(off + 4);
          cmd.arg1         = read64(off + 4 + 8);
          cmd.arg2         = read64(off + 4 + 16);
          cmd.profile_slot = profiled ? read64(off + int'(sz) - 8) : 64'd0;
          cmd_size         = OFF_W'(sz);
          has_cmd          = 1'b1;
        end
      end
    end
  end

endmodule : VX_cp_unpack
