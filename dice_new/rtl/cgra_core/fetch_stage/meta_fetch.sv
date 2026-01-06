`include "dice_define.vh"
`include "VX_define.vh"

module meta_fetch
  import dice_pkg::*;
  import dice_frontend_pkg::*;
#(
    parameter int TAG_WIDTH = 48
) (
    input logic clk_i,
    input logic rst_i,

    // From CS/FDR barrier
    input logic                                              schedule_valid_i,
    input  logic [DICE_ADDR_WIDTH-1:0]      fdr_next_pc_i,
    input  logic [EBLOCK_ID_WIDTH-1:0]                schedule_eblock_id_i,
    output logic                                             schedule_ready_o,

    // Request channel to cache
    VX_mem_bus_if.master meta_fetch_bus_if,

    // To decoder
    output pgraph_meta_t                    outgoing_meta_o,
    output logic                            meta_valid_o,

    // From stage barrier
    input logic fire_eblock_i
);
  localparam int PadWidth = TAG_WIDTH - EBLOCK_ID_WIDTH;

  // FSM states
  typedef enum logic [1:0] {
    StateReady    = 2'b00,  // fetcher is ready for a new pc
    StateReqVal   = 2'b01,
    StateWaitResp = 2'b10,  // waiting for response from cache
    StateHoldData = 2'b11   // waits for decoder to consume meta
  } meta_fetch_state_e;

  meta_fetch_state_e state_q, state_d;
  logic meta_valid_q;
  logic [EBLOCK_ID_WIDTH-1:0] eblock_id_q;
  pgraph_meta_t outgoing_meta_q;

  //DIRECTLY FROM VORTEX======================================================
  logic [VX_gpu_pkg::ICACHE_ADDR_WIDTH-1:0] meta_cache_req_addr_q, meta_cache_req_addr_d;
  localparam int AddrShift = $clog2(
      VX_gpu_pkg::VX_MEM_DATA_WIDTH / 8
  );
  assign meta_cache_req_addr_d = fdr_next_pc_i[AddrShift+:VX_gpu_pkg::ICACHE_ADDR_WIDTH];
  // 4-byte aligned addresses
  //DIRECTLY FROM VORTEX======================================================

  logic rsp_fire, req_fire;
  assign rsp_fire = meta_fetch_bus_if.rsp_valid && meta_fetch_bus_if.rsp_ready;
  assign req_fire = meta_fetch_bus_if.req_valid && meta_fetch_bus_if.req_ready;

  always_comb begin
    // Default assignments at top of always_comb
    schedule_ready_o = 1'b0;
    state_d          = state_q;

    unique case (state_q)
      StateReady: begin
        schedule_ready_o = 1'b1;
        if (schedule_valid_i) begin
          state_d = StateReqVal;
        end
      end
      StateReqVal: begin
        if (req_fire) state_d = StateWaitResp;
      end
      StateWaitResp: begin
        if (rsp_fire) begin
          state_d = StateHoldData;
        end
      end
      StateHoldData: begin
        if (fire_eblock_i) state_d = StateReady;
      end
      default: state_d = StateReady;
    endcase
  end


  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q               <= StateReady;
      meta_valid_q          <= 1'b0;
      meta_cache_req_addr_q <= '0;
      outgoing_meta_q       <= '0;
      eblock_id_q           <= '0;
    end else begin
      state_q <= state_d;
      if (state_q == StateReady && schedule_valid_i && schedule_ready_o) begin
        meta_cache_req_addr_q <= meta_cache_req_addr_d;
        eblock_id_q           <= schedule_eblock_id_i;
      end
      if (rsp_fire) begin
        outgoing_meta_q <= pgraph_meta_t'(
            meta_fetch_bus_if.rsp_data.data);
        meta_valid_q    <= 1'b1;
      end
      if (fire_eblock_i) begin
        meta_valid_q <= 1'b0;
      end
    end
  end


  //============= UNUSED VORTEX CACHE FEATURES =================//
  assign meta_fetch_bus_if.req_data.flags  = '0;  //misc / not used
  assign meta_fetch_bus_if.req_data.rw     = 0;   //read/write bit
  assign meta_fetch_bus_if.req_data.byteen = '1;  //byte mask (for stores)
  assign meta_fetch_bus_if.req_data.data   = '0;  //write payload

  //Pad unused part of VORTEX tag with zeros
  //Logic is less complicated since there is only one cta
  //in fdr at once as opposed to vortexs large # of warps
  assign meta_fetch_bus_if.req_data.tag    = {{PadWidth{1'b0}}, eblock_id_q};

  //============ MISC ASSIGNS ======================//
  assign meta_fetch_bus_if.req_data.addr   = meta_cache_req_addr_q;
  assign meta_fetch_bus_if.req_valid       = (state_q == StateReqVal);
  assign meta_fetch_bus_if.rsp_ready       = (state_q == StateWaitResp);
  assign meta_valid_o                      = meta_valid_q;
  assign outgoing_meta_o                   = outgoing_meta_q;
endmodule
