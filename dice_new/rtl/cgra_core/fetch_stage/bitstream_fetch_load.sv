`include "VX_define.vh"

module bitstream_fetch_load
  import dice_pkg::*;
  import dice_frontend_pkg::*;
#(
    parameter int TAG_WIDTH = 48,
    parameter int BITSTREAM_SIZE = 2056,
    parameter int CHUNK_SIZE = VX_gpu_pkg::VX_MEM_DATA_WIDTH,
    parameter int NUM_CHUNKS = (BITSTREAM_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE
) (
    input logic clk_i,
    input logic rst_i,

    //from decoder
    input logic                                 meta_valid_i,
    input logic [DICE_ADDR_WIDTH-1:0] bitstream_addr_i,

    //to cgra buffers
    output logic [CHUNK_SIZE-1:0]  cm0_data_o,
    output logic [NUM_CHUNKS-1:0]  cm0_chunk_en_o,

    output logic [CHUNK_SIZE-1:0]  cm1_data_o,
    output logic [NUM_CHUNKS-1:0]  cm1_chunk_en_o,

    //to valid checker
    output logic done_streaming_o,

    //cache interface
    VX_mem_bus_if.master cache_bus_if,

    //to FDR EX buffer
    output logic cm_num_o
);

  localparam int CounterBits = $clog2(NUM_CHUNKS + 1);
  localparam int Offset = CHUNK_SIZE / 8;

  typedef enum logic [1:0] {
    StateIdle,
    StateStreaming,  // Handles both Request and Response phases
    StateDone
  } bitstream_fetch_state_e;

  bitstream_fetch_state_e state_q, state_d;

  // registered states
  logic [DICE_ADDR_WIDTH-1:0] cm0_addr_q, cm1_addr_q, cm0_addr_d, cm1_addr_d;
  logic cm_select_q, cm_select_d;  // 0 = cm0, 1 = cm1

  logic [CounterBits-1:0] chunk_count_q, chunk_count_d;  //how many chunks have been streamed

  // Data
  logic [CHUNK_SIZE-1:0] data_chunk_q, data_chunk_d;

  logic [DICE_ADDR_WIDTH-1:0] addr_q, addr_d;
  logic cm0_valid_d, cm1_valid_d, cm0_valid_q, cm1_valid_q;

  // Track if we have sent the request for the current chunk
  logic req_sent_q, req_sent_d;

  logic [NUM_CHUNKS-1:0] load_chunk_en_d;
  logic [NUM_CHUNKS-1:0] load_chunk_en_q;

  // Address alias
  logic [DICE_ADDR_WIDTH-1:0] bitstream_addr_dec;
  assign bitstream_addr_dec = bitstream_addr_i;

  // Bus Handshake Signals
  logic req_fire;
  logic rsp_fire;

  assign req_fire = cache_bus_if.req_valid && cache_bus_if.req_ready;
  assign rsp_fire = cache_bus_if.rsp_valid && cache_bus_if.rsp_ready;

  // Vortex Bus Assignments
  assign cache_bus_if.req_data.flags = '0;
  assign cache_bus_if.req_data.rw = 0;  // Read
  assign cache_bus_if.req_data.byteen = '1;  // All bytes enabled
  assign cache_bus_if.req_data.data = '0;

  assign cache_bus_if.req_data.tag = {TAG_WIDTH{1'b0}};

  assign cache_bus_if.req_data.addr = addr_q;

  // We are valid to request if we are streaming and haven't sent the request yet
  assign cache_bus_if.req_valid = (state_q == StateStreaming) && !req_sent_q;

  // We are ready for a response if we are streaming and HAVE sent the request
  assign cache_bus_if.rsp_ready = (state_q == StateStreaming) && req_sent_q;

  // Output assignments
  assign cm_num_o = cm_select_q;
  assign cm0_data_o = data_chunk_q;
  assign cm1_data_o = data_chunk_q;

  assign done_streaming_o = (cm_select_q == 1'b0 && cm0_valid_q &&
                             cm0_addr_q == bitstream_addr_dec) ||
                            (cm_select_q == 1'b1 && cm1_valid_q &&
                             cm1_addr_q == bitstream_addr_dec);

  always_comb begin
    // Default assignments at top of always_comb
    state_d         = state_q;
    chunk_count_d   = chunk_count_q;
    cm_select_d     = cm_select_q;
    cm0_addr_d      = cm0_addr_q;
    cm1_addr_d      = cm1_addr_q;
    data_chunk_d    = data_chunk_q;
    addr_d          = addr_q;
    cm0_valid_d     = cm0_valid_q;
    cm1_valid_d     = cm1_valid_q;
    req_sent_d      = req_sent_q;
    load_chunk_en_d = '0;
    cm0_chunk_en_o  = '0;
    cm1_chunk_en_o  = '0;

    // Output chunk enables based on select
    if (cm_select_q == 1'b0) begin
      cm0_chunk_en_o = load_chunk_en_q;
    end else begin
      cm1_chunk_en_o = load_chunk_en_q;
    end

    unique case (state_q)
      StateIdle: begin
        req_sent_d = 1'b0;
        if (meta_valid_i) begin
          if (!done_streaming_o) begin
            if (cm_select_q == 1'b0 && cm1_valid_q && cm1_addr_q == bitstream_addr_dec) begin
              cm_select_d = 1'b1;
            end else if (cm_select_q == 1'b1 && cm0_valid_q &&
                         cm0_addr_q == bitstream_addr_dec) begin
              cm_select_d = 1'b0;
            end else begin
              if (cm0_valid_q || cm1_valid_q) cm_select_d = ~cm_select_q;
              else cm_select_d = 1'b0;

              addr_d = bitstream_addr_dec;
              state_d = StateStreaming;
              chunk_count_d = '0;

              if (cm_select_d == 1'b0) begin
                cm0_addr_d  = bitstream_addr_dec;
                cm0_valid_d = 1'b0;
              end else begin
                cm1_addr_d  = bitstream_addr_dec;
                cm1_valid_d = 1'b0;
              end
            end
          end
        end
      end
      StateStreaming: begin
        if (!req_sent_q) begin
          if (req_fire) begin
            req_sent_d = 1'b1;
          end
        end else begin
          if (rsp_fire) begin
            data_chunk_d = cache_bus_if.rsp_data.data;
            load_chunk_en_d = (1'b1 << chunk_count_q);
            chunk_count_d = chunk_count_q + 1'b1;
            req_sent_d = 1'b0;
            if (chunk_count_q == NUM_CHUNKS - 1) begin
              state_d = StateDone;
            end else begin
              addr_d = addr_q + Offset;
            end
          end
        end
      end

      StateDone: begin
        state_d = StateIdle;
        if (cm_select_q == 1'b1) begin
          cm1_valid_d = 1'b1;
        end else begin
          cm0_valid_d = 1'b1;
        end
      end
      default: begin
        state_d = StateIdle;
      end
    endcase

  end

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q         <= StateIdle;
      chunk_count_q   <= '0;
      cm_select_q     <= 1'b0;
      data_chunk_q    <= '0;
      cm0_addr_q      <= '0;
      cm1_addr_q      <= '0;
      addr_q          <= '0;
      cm0_valid_q     <= 1'b0;
      cm1_valid_q     <= 1'b0;
      load_chunk_en_q <= '0;
      req_sent_q      <= 1'b0;
    end else begin
      state_q         <= state_d;
      chunk_count_q   <= chunk_count_d;
      cm0_addr_q      <= cm0_addr_d;
      cm1_addr_q      <= cm1_addr_d;
      data_chunk_q    <= data_chunk_d;
      cm_select_q     <= cm_select_d;
      addr_q          <= addr_d;
      cm0_valid_q     <= cm0_valid_d;
      cm1_valid_q     <= cm1_valid_d;
      load_chunk_en_q <= load_chunk_en_d;
      req_sent_q      <= req_sent_d;
    end
  end
endmodule
