// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifndef NOPAE
`include "afu_json_info.vh"
`else
`include "vortex_opae.vh"
`endif

// ============================================================================
// OPAE AFU shim — thin adapter. The Command Processor is the sole command
// path and the sole DMA engine:
//
//   * MMIO  — CCI-P c0 MMIO. Host byte 0x000-page = the AFU Device Feature
//             Header (required by OPAE) + the SCOPE register pair. Host byte
//             0x1000+ (mmio address bit 10) = the CP regfile (cp_axil).
//   * Device memory — Vortex's banks + the CP's axi_dev master share the
//             Avalon local-memory subsystem through VX_mem_arb.
//   * Host memory — the CP's axi_host master reaches host memory over CCI-P
//             (c0 reads / c1 writes) via VX_cp_axi_to_membus + a small
//             CCI-P bridge. This is the only user of CCI-P c0/c1, and the
//             only host<->device DMA on the platform.
//
// The legacy STATE_* command FSM, the legacy CCI-P DMA engine, and the
// legacy MMIO command/caps/DCR surface were removed — the CP replaces them.
// ============================================================================

module vortex_afu import ccip_if_pkg::*; import local_mem_cfg_pkg::*; import VX_gpu_pkg::*; #(
    parameter NUM_LOCAL_MEM_BANKS = 2
) (
    // global signals
    input wire clk,
    input wire reset,

    // IF signals between CCI and AFU
    input   t_if_ccip_Rx  cp2af_sRxPort,
    output  t_if_ccip_Tx  af2cp_sTxPort,

    // Avalon signals for local memory access
    output  t_local_mem_data      avs_writedata [NUM_LOCAL_MEM_BANKS],
    input   t_local_mem_data      avs_readdata [NUM_LOCAL_MEM_BANKS],
    output  t_local_mem_addr      avs_address [NUM_LOCAL_MEM_BANKS],
    input   wire                  avs_waitrequest [NUM_LOCAL_MEM_BANKS],
    output  wire                  avs_write [NUM_LOCAL_MEM_BANKS],
    output  wire                  avs_read [NUM_LOCAL_MEM_BANKS],
    output  t_local_mem_byte_mask avs_byteenable [NUM_LOCAL_MEM_BANKS],
    output  t_local_mem_burst_cnt avs_burstcount [NUM_LOCAL_MEM_BANKS],
    input   wire                  avs_readdatavalid [NUM_LOCAL_MEM_BANKS]
);
    localparam LMEM_DATA_WIDTH    = $bits(t_local_mem_data);
    localparam LMEM_DATA_SIZE     = LMEM_DATA_WIDTH / 8;
    localparam LMEM_ADDR_WIDTH    = $bits(t_local_mem_addr);
    localparam LMEM_BURST_CTRW    = $bits(t_local_mem_burst_cnt);

    localparam CCI_VX_ADDR_WIDTH  = VX_MEM_ADDR_WIDTH + ($clog2(VX_MEM_DATA_WIDTH) - $clog2(LMEM_DATA_WIDTH));

    localparam CCI_DATA_WIDTH     = $bits(t_ccip_clData);
    localparam CCI_ADDR_WIDTH     = $bits(t_ccip_clAddr);

    localparam AVS_RD_QUEUE_SIZE  = 32;
    localparam VX_AVS_REQ_TAGW    = VX_MEM_TAG_WIDTH + `CLOG2(LMEM_DATA_WIDTH) - `CLOG2(VX_MEM_DATA_WIDTH);
    localparam VX_AVS_REQ_TAGW2   = `MAX(VX_MEM_TAG_WIDTH, VX_AVS_REQ_TAGW);
    localparam CCI_VX_TAG_WIDTH   = `MAX(VX_AVS_REQ_TAGW2, CCI_ADDR_WIDTH);
    localparam AVS_TAG_WIDTH      = CCI_VX_TAG_WIDTH + 1; // 1 arbiter bit (2 inputs: Vortex + CP)

    localparam AFU_ID_L           = 16'h0002;      // AFU ID Lower
    localparam AFU_ID_H           = 16'h0004;      // AFU ID Higher

    wire [127:0] afu_id = `AFU_ACCEL_UUID;

    localparam CLUSTER_SIZE = `VX_CFG_NUM_CORES / `VX_CFG_SOCKET_SIZE;
    `STATIC_ASSERT((CLUSTER_SIZE * `VX_CFG_SOCKET_SIZE) == `VX_CFG_NUM_CORES, ("NUM_CORES must be a multiple of SOCKET_SIZE"));

    // Vortex memory ports ////////////////////////////////////////////////////

    wire                            vx_mem_req_valid [VX_MEM_PORTS];
    wire                            vx_mem_req_rw [VX_MEM_PORTS];
    wire [VX_MEM_BYTEEN_WIDTH-1:0]  vx_mem_req_byteen [VX_MEM_PORTS];
    wire [VX_MEM_ADDR_WIDTH-1:0]    vx_mem_req_addr [VX_MEM_PORTS];
    wire [VX_MEM_DATA_WIDTH-1:0]    vx_mem_req_data [VX_MEM_PORTS];
    wire [VX_MEM_TAG_WIDTH-1:0]     vx_mem_req_tag [VX_MEM_PORTS];
    wire                            vx_mem_req_ready [VX_MEM_PORTS];

    wire                            vx_mem_rsp_valid [VX_MEM_PORTS];
    wire [VX_MEM_DATA_WIDTH-1:0]    vx_mem_rsp_data [VX_MEM_PORTS];
    wire [VX_MEM_TAG_WIDTH-1:0]     vx_mem_rsp_tag [VX_MEM_PORTS];
    wire                            vx_mem_rsp_ready [VX_MEM_PORTS];

    // MMIO controller ////////////////////////////////////////////////////////

    t_ccip_c0_ReqMmioHdr mmio_req_hdr;
    assign mmio_req_hdr = t_ccip_c0_ReqMmioHdr'(cp2af_sRxPort.c0.hdr[$bits(t_ccip_c0_ReqMmioHdr)-1:0]);
    `UNUSED_VAR (mmio_req_hdr)

    t_if_ccip_c2_Tx mmio_rsp;

    // MMIO response mux: the DFH handler drives `mmio_rsp` on the next cycle
    // for non-CP reads; the CP regfile drives `cp_mmio_rsp` on its slave's
    // rvalid pulse. They never fire together — the DFH handler is gated on
    // `!is_cp_mmio_req`.
    t_if_ccip_c2_Tx cp_mmio_rsp;
    assign af2cp_sTxPort.c2 = cp_mmio_rsp.mmioRdValid ? cp_mmio_rsp : mmio_rsp;

    // ========================================================================
    // Command Processor MMIO demux. mmio_req_hdr.address is in 4-byte units;
    // bit 10 (= 0x400) corresponds to host byte address 0x1000.
    //   host byte 0x000..0xFFF  (address[10]=0) -> AFU DFH / SCOPE handler
    //   host byte 0x1000+       (address[10]=1) -> CP regfile (cp_axil)
    // ========================================================================
    wire is_cp_mmio_req = mmio_req_hdr.address[10];
    wire cp_mmio_wr     = cp2af_sRxPort.c0.mmioWrValid && is_cp_mmio_req;
    wire cp_mmio_rd     = cp2af_sRxPort.c0.mmioRdValid && is_cp_mmio_req;

    VX_cp_axil_s_if #(.ADDR_W(16)) cp_axil ();

    // CCIP packs AW + W into one mmioWrValid pulse — present them together
    // to the AXI-Lite slave. Truncate the host's 64-bit data to the low 32
    // bits (every CP register is 32-bit).
    assign cp_axil.awvalid = cp_mmio_wr;
    assign cp_axil.awaddr  = {4'd0, mmio_req_hdr.address[9:0], 2'd0};
    assign cp_axil.wvalid  = cp_mmio_wr;
    assign cp_axil.wdata   = cp2af_sRxPort.c0.data[31:0];
    assign cp_axil.wstrb   = 4'hF;
    assign cp_axil.bready  = 1'b1;                 // CCIP has no B channel
    `UNUSED_VAR (cp_axil.bvalid)
    `UNUSED_VAR (cp_axil.bresp)

    assign cp_axil.arvalid = cp_mmio_rd;
    assign cp_axil.araddr  = {4'd0, mmio_req_hdr.address[9:0], 2'd0};
    assign cp_axil.rready  = 1'b1;
    `UNUSED_VAR (cp_axil.rresp)

    // Latch the read tid when a CP read fires; present the CP regfile's
    // rdata on the CCIP response channel when its rvalid arrives.
    reg        cp_rd_pending;
    t_ccip_tid cp_rd_tid;
    always @(posedge clk) begin
        if (reset) begin
            cp_rd_pending <= 1'b0;
            cp_rd_tid     <= '0;
        end else begin
            if (cp_mmio_rd) begin
                cp_rd_pending <= 1'b1;
                cp_rd_tid     <= mmio_req_hdr.tid;
            end else if (cp_axil.rvalid) begin
                cp_rd_pending <= 1'b0;
            end
        end
    end
    `UNUSED_VAR (cp_rd_pending)

    always @(*) begin
        cp_mmio_rsp = '0;
        if (cp_axil.rvalid) begin
            cp_mmio_rsp.mmioRdValid = 1'b1;
            cp_mmio_rsp.hdr.tid     = cp_rd_tid;
            cp_mmio_rsp.data        = 64'(cp_axil.rdata);
        end
    end

`ifdef SCOPE

    localparam MMIO_SCOPE_READ  = `AFU_IMAGE_MMIO_SCOPE_READ;
    localparam MMIO_SCOPE_WRITE = `AFU_IMAGE_MMIO_SCOPE_WRITE;

    reg [63:0] cmd_scope_rdata;
    reg [63:0] cmd_scope_wdata;
    reg cmd_scope_reading;
    reg cmd_scope_writing;
    reg  scope_bus_in;
    wire scope_bus_out;
    reg [5:0] scope_bus_ctr;
    wire scope_reset = reset;

    always @(posedge clk) begin
        if (reset) begin
            cmd_scope_reading <= 0;
            cmd_scope_writing <= 0;
            scope_bus_in      <= 0;
        end else begin
            scope_bus_in <= 0;
            if (scope_bus_out) begin
                cmd_scope_reading <= 1;
                scope_bus_ctr     <= 63;
            end
            if (cp2af_sRxPort.c0.mmioWrValid
             && (MMIO_SCOPE_WRITE == mmio_req_hdr.address)) begin
                cmd_scope_wdata   <= 64'(cp2af_sRxPort.c0.data);
                cmd_scope_writing <= 1;
                scope_bus_ctr     <= 63;
                scope_bus_in      <= 1;
            end
            if (cmd_scope_writing) begin
                scope_bus_in  <= cmd_scope_wdata[scope_bus_ctr];
                scope_bus_ctr <= scope_bus_ctr - 6'd1;
                if (scope_bus_ctr == 0) begin
                    cmd_scope_writing <= 0;
                    scope_bus_ctr <= 0;
                end
            end
            if (cmd_scope_reading) begin
                cmd_scope_rdata <= {cmd_scope_rdata[62:0], scope_bus_out};
                scope_bus_ctr   <= scope_bus_ctr - 6'd1;
                if (scope_bus_ctr == 0) begin
                    cmd_scope_reading <= 0;
                    scope_bus_ctr <= 0;
                end
            end
        end
    end

`endif

`ifdef SIMULATION
`ifndef VERILATOR
    reg [`CLOG2(`VX_CFG_RESET_DELAY+1)-1:0] assert_delay_ctr;
    initial begin
        $assertoff;
    end
    always @(posedge clk) begin
        if (reset) begin
            assert_delay_ctr <= '0;
        end else begin
            assert_delay_ctr <= assert_delay_ctr + $bits(assert_delay_ctr)'(1);
            if (assert_delay_ctr == (`VX_CFG_RESET_DELAY-1)) begin
                $asserton;
            end
        end
    end
`endif
`endif

    // Handle MMIO read requests — AFU Device Feature Header + SCOPE. The CP
    // range is answered via the cp_mmio_rsp path, so suppress the DFH
    // response for those addresses.
    always @(posedge clk) begin
        if (reset) begin
            mmio_rsp.mmioRdValid <= 0;
        end else begin
            mmio_rsp.mmioRdValid <= cp2af_sRxPort.c0.mmioRdValid && !is_cp_mmio_req;
        end

        mmio_rsp.hdr.tid <= mmio_req_hdr.tid;

        if (cp2af_sRxPort.c0.mmioRdValid) begin
            case (mmio_req_hdr.address)
            // AFU header
            16'h0000: mmio_rsp.data <= {
                4'b0001, // Feature type = AFU
                8'b0,    // reserved
                4'b0,    // afu minor revision = 0
                7'b0,    // reserved
                1'b1,    // end of DFH list = 1
                24'b0,   // next DFH offset = 0
                4'b0,    // afu major revision = 0
                12'b0    // feature ID = 0
            };
            AFU_ID_L: mmio_rsp.data <= afu_id[63:0];   // afu id low
            AFU_ID_H: mmio_rsp.data <= afu_id[127:64]; // afu id hi
            16'h0006: mmio_rsp.data <= 64'h0; // next AFU
            16'h0008: mmio_rsp.data <= 64'h0; // reserved
        `ifdef SCOPE
            MMIO_SCOPE_READ: begin
                mmio_rsp.data <= cmd_scope_rdata;
            end
        `endif
            default: begin
                mmio_rsp.data <= 64'h0;
            `ifdef DBG_TRACE_AFU
                `TRACE(2, ("%t: AFU: Unknown MMIO Rd: addr=0x%0h\n", $time, mmio_req_hdr.address))
            `endif
            end
            endcase
        end
    end

`ifdef DBG_TRACE_AFU
    always @(posedge clk) begin
        if (cp2af_sRxPort.c0.mmioWrValid && !is_cp_mmio_req) begin
            `TRACE(2, ("%t: AFU: MMIO Wr: addr=0x%0h, data=0x%h\n", $time, mmio_req_hdr.address, 64'(cp2af_sRxPort.c0.data)))
        end
    end
`endif

    // Vortex reset shift ////////////////////////////////////////////////////

    reg [`VX_CFG_RESET_DELAY-1:0] vx_reset_shift_r;
    wire vx_reset;
    wire vx_start;
    wire vx_busy;

    initial begin
        vx_reset_shift_r = {`VX_CFG_RESET_DELAY{1'b1}};
    end
    assign vx_reset = vx_reset_shift_r[`VX_CFG_RESET_DELAY-1];

    always @(posedge clk) begin
        if (reset) begin
            vx_reset_shift_r <= {`VX_CFG_RESET_DELAY{1'b1}};
        end else begin
            vx_reset_shift_r <= {vx_reset_shift_r[`VX_CFG_RESET_DELAY-2:0], 1'b0};
        end
    end

    // Command Processor //////////////////////////////////////////////////////

    VX_cp_gpu_if cp_gpu_if ();
    VX_cp_axi_m_if #(.ADDR_W(64), .DATA_W(LMEM_DATA_WIDTH)) cp_axi_dev  ();
    VX_cp_axi_m_if #(.ADDR_W(64), .DATA_W(CCI_DATA_WIDTH))  cp_axi_host ();

    // The CCI-P AFU has no dedicated platform interrupt pin — the CP
    // interrupt stays unconsumed here.
    wire cp_interrupt;
    `UNUSED_VAR (cp_interrupt)

    VX_cp_core u_cp_core (
        .clk        (clk),
        .reset      (reset),
        .axil_s     (cp_axil),
        .axi_host   (cp_axi_host),
        .axi_dev    (cp_axi_dev),
        .gpu_if     (cp_gpu_if),
        .irq        (cp_interrupt)
    );

    // The CP is the sole launch + DCR source.
    assign vx_start = cp_gpu_if.start;
    assign cp_gpu_if.busy = vx_busy;

    wire                         vx_dcr_req_valid = cp_gpu_if.dcr_req_valid;
    wire                         vx_dcr_req_rw    = cp_gpu_if.dcr_req_rw;
    wire [VX_DCR_ADDR_WIDTH-1:0] vx_dcr_req_addr  = cp_gpu_if.dcr_req_addr;
    wire [VX_DCR_DATA_WIDTH-1:0] vx_dcr_req_data  = cp_gpu_if.dcr_req_data;
    wire                         vx_dcr_rsp_valid;
    wire [VX_DCR_DATA_WIDTH-1:0] vx_dcr_rsp_data;

    assign cp_gpu_if.dcr_req_ready = 1'b1;          // Vortex DCR always accepts
    assign cp_gpu_if.dcr_rsp_valid = vx_dcr_rsp_valid;
    assign cp_gpu_if.dcr_rsp_data  = vx_dcr_rsp_data;

    // ========================================================================
    // CP host-memory bridge — axi_host -> VX_cp_axi_to_membus -> CCI-P.
    // Single outstanding request at a time; the CP fetches/DMA-stages one
    // cache line per CCI-P transaction.
    // ========================================================================
    localparam HB_ADDR_W = 64 - $clog2(CCI_DATA_WIDTH/8);

    wire                            hb_req_valid;
    wire                            hb_req_rw;
    wire [HB_ADDR_W-1:0]            hb_req_addr;
    wire [CCI_DATA_WIDTH-1:0]       hb_req_data;
    wire [CCI_DATA_WIDTH/8-1:0]     hb_req_byteen;
    wire [`VX_CP_AXI_TID_WIDTH-1:0] hb_req_tag;
    wire                            hb_req_ready;
    wire                            hb_rsp_valid;
    wire [CCI_DATA_WIDTH-1:0]       hb_rsp_data;
    wire [`VX_CP_AXI_TID_WIDTH-1:0] hb_rsp_tag;
    wire                            hb_rsp_ready;

    VX_cp_axi_to_membus #(
        .ADDR_W (64),
        .DATA_W (CCI_DATA_WIDTH),
        .ID_W   (`VX_CP_AXI_TID_WIDTH)
    ) u_cp_host_bridge (
        .clk            (clk),
        .reset          (reset),
        .axi_s          (cp_axi_host),
        .mem_req_valid  (hb_req_valid),
        .mem_req_rw     (hb_req_rw),
        .mem_req_addr   (hb_req_addr),
        .mem_req_data   (hb_req_data),
        .mem_req_byteen (hb_req_byteen),
        .mem_req_tag    (hb_req_tag),
        .mem_req_ready  (hb_req_ready),
        .mem_rsp_valid  (hb_rsp_valid),
        .mem_rsp_data   (hb_rsp_data),
        .mem_rsp_tag    (hb_rsp_tag),
        .mem_rsp_ready  (hb_rsp_ready)
    );
    `UNUSED_VAR (hb_req_byteen)   // CCI-P writes whole cache lines
    `UNUSED_VAR (hb_req_tag)
    `UNUSED_VAR (hb_req_addr)     // high bits beyond t_ccip_clAddr are zero

    localparam [1:0] HB_IDLE = 2'd0, HB_RD = 2'd1, HB_RD_RSP = 2'd2, HB_WR = 2'd3;
    reg [1:0]                hb_state;
    reg [CCI_DATA_WIDTH-1:0] hb_data_r;

    wire hb_c0_rsp = cp2af_sRxPort.c0.rspValid
                  && (cp2af_sRxPort.c0.hdr.resp_type == eRSP_RDLINE);
    wire hb_c1_rsp = cp2af_sRxPort.c1.rspValid
                  && (cp2af_sRxPort.c1.hdr.resp_type == eRSP_WRLINE);

    wire hb_rd_go = (hb_state == HB_IDLE) && hb_req_valid && !hb_req_rw
                 && !cp2af_sRxPort.c0TxAlmFull;
    wire hb_wr_go = (hb_state == HB_IDLE) && hb_req_valid && hb_req_rw
                 && !cp2af_sRxPort.c1TxAlmFull;

    always @(posedge clk) begin
        if (reset) begin
            hb_state  <= HB_IDLE;
            hb_data_r <= '0;
        end else begin
            case (hb_state)
            HB_IDLE: begin
                if (hb_rd_go)      hb_state <= HB_RD;
                else if (hb_wr_go) hb_state <= HB_WR;
            end
            HB_RD: begin
                if (hb_c0_rsp) begin
                    hb_data_r <= cp2af_sRxPort.c0.data;
                    hb_state  <= HB_RD_RSP;
                end
            end
            HB_RD_RSP: begin
                if (hb_rsp_ready) hb_state <= HB_IDLE;
            end
            HB_WR: begin
                if (hb_c1_rsp) hb_state <= HB_IDLE;
            end
            default: hb_state <= HB_IDLE;
            endcase
        end
    end

    // membus handshake back to VX_cp_axi_to_membus.
    assign hb_req_ready = hb_rd_go || (hb_state == HB_WR && hb_c1_rsp);
    assign hb_rsp_valid = (hb_state == HB_RD_RSP);
    assign hb_rsp_data  = hb_data_r;
    assign hb_rsp_tag   = '0;

    // CCI-P TX channels c0 (reads) / c1 (writes).
    always @(*) begin
        af2cp_sTxPort.c0         = '0;
        af2cp_sTxPort.c0.valid   = hb_rd_go;
        af2cp_sTxPort.c0.hdr     = t_ccip_c0_ReqMemHdr'(0);
        af2cp_sTxPort.c0.hdr.address = t_ccip_clAddr'(hb_req_addr);

        af2cp_sTxPort.c1         = '0;
        af2cp_sTxPort.c1.valid   = hb_wr_go;
        af2cp_sTxPort.c1.hdr     = t_ccip_c1_ReqMemHdr'(0);
        af2cp_sTxPort.c1.hdr.sop = 1'b1;            // single-line write
        af2cp_sTxPort.c1.hdr.address = t_ccip_clAddr'(hb_req_addr);
        af2cp_sTxPort.c1.data    = t_ccip_clData'(hb_req_data);
    end

    // Device memory subsystem ////////////////////////////////////////////////
    // Vortex's bank-0 port and the CP's axi_dev master share local memory
    // through a 2-input arbiter; banks 1..N-1 pass straight through.

    VX_mem_bus_if #(
        .DATA_SIZE  (LMEM_DATA_SIZE),
        .ADDR_WIDTH (CCI_VX_ADDR_WIDTH),
        .TAG_WIDTH  (CCI_VX_TAG_WIDTH)
    ) vx_mem_bus_if[VX_MEM_PORTS]();

    for (genvar i = 0; i < VX_MEM_PORTS; ++i) begin : g_vx_mem_adapter
        VX_mem_data_adapter #(
            .SRC_DATA_WIDTH (VX_MEM_DATA_WIDTH),
            .DST_DATA_WIDTH (LMEM_DATA_WIDTH),
            .SRC_ADDR_WIDTH (VX_MEM_ADDR_WIDTH),
            .DST_ADDR_WIDTH (CCI_VX_ADDR_WIDTH),
            .SRC_TAG_WIDTH  (VX_MEM_TAG_WIDTH),
            .DST_TAG_WIDTH  (CCI_VX_TAG_WIDTH),
            .REQ_OUT_BUF    (0),
            .RSP_OUT_BUF    (2)
        ) vx_mem_data_adapter (
            .clk                (clk),
            .reset              (reset),

            .mem_req_valid_in   (vx_mem_req_valid[i]),
            .mem_req_addr_in    (vx_mem_req_addr[i]),
            .mem_req_rw_in      (vx_mem_req_rw[i]),
            .mem_req_byteen_in  (vx_mem_req_byteen[i]),
            .mem_req_data_in    (vx_mem_req_data[i]),
            .mem_req_tag_in     (vx_mem_req_tag[i]),
            .mem_req_ready_in   (vx_mem_req_ready[i]),

            .mem_rsp_valid_in   (vx_mem_rsp_valid[i]),
            .mem_rsp_data_in    (vx_mem_rsp_data[i]),
            .mem_rsp_tag_in     (vx_mem_rsp_tag[i]),
            .mem_rsp_ready_in   (vx_mem_rsp_ready[i]),

            .mem_req_valid_out  (vx_mem_bus_if[i].req_valid),
            .mem_req_addr_out   (vx_mem_bus_if[i].req_data.addr),
            .mem_req_rw_out     (vx_mem_bus_if[i].req_data.rw),
            .mem_req_byteen_out (vx_mem_bus_if[i].req_data.byteen),
            .mem_req_data_out   (vx_mem_bus_if[i].req_data.data),
            .mem_req_tag_out    (vx_mem_bus_if[i].req_data.tag),
            .mem_req_ready_out  (vx_mem_bus_if[i].req_ready),

            .mem_rsp_valid_out  (vx_mem_bus_if[i].rsp_valid),
            .mem_rsp_data_out   (vx_mem_bus_if[i].rsp_data.data),
            .mem_rsp_tag_out    (vx_mem_bus_if[i].rsp_data.tag),
            .mem_rsp_ready_out  (vx_mem_bus_if[i].rsp_ready)
        );
        assign vx_mem_bus_if[i].req_data.attr = '0;
    end

    // CP axi_dev -> VX_mem_bus bridge.
    VX_mem_bus_if #(
        .DATA_SIZE  (LMEM_DATA_SIZE),
        .ADDR_WIDTH (CCI_VX_ADDR_WIDTH),
        .TAG_WIDTH  (CCI_VX_TAG_WIDTH)
    ) cp_vx_mem_arb_in_if[2]();   // [0] = Vortex bank0, [1] = CP axi_dev

    wire                              cp_membus_req_valid;
    wire                              cp_membus_req_rw;
    wire [64 - $clog2(LMEM_DATA_WIDTH/8) - 1:0] cp_membus_req_addr_full;
    wire [LMEM_DATA_WIDTH-1:0]        cp_membus_req_data;
    wire [LMEM_DATA_WIDTH/8-1:0]      cp_membus_req_byteen;
    wire [`VX_CP_AXI_TID_WIDTH-1:0]   cp_membus_req_tag;
    wire                              cp_membus_req_ready;
    wire                              cp_membus_rsp_valid;
    wire [LMEM_DATA_WIDTH-1:0]        cp_membus_rsp_data;
    wire [`VX_CP_AXI_TID_WIDTH-1:0]   cp_membus_rsp_tag;
    wire                              cp_membus_rsp_ready;

    VX_cp_axi_to_membus #(
        .ADDR_W   (64),
        .DATA_W   (LMEM_DATA_WIDTH),
        .ID_W     (`VX_CP_AXI_TID_WIDTH)
    ) u_cp_dev_bridge (
        .clk            (clk),
        .reset          (reset),
        .axi_s          (cp_axi_dev),
        .mem_req_valid  (cp_membus_req_valid),
        .mem_req_rw     (cp_membus_req_rw),
        .mem_req_addr   (cp_membus_req_addr_full),
        .mem_req_data   (cp_membus_req_data),
        .mem_req_byteen (cp_membus_req_byteen),
        .mem_req_tag    (cp_membus_req_tag),
        .mem_req_ready  (cp_membus_req_ready),
        .mem_rsp_valid  (cp_membus_rsp_valid),
        .mem_rsp_data   (cp_membus_rsp_data),
        .mem_rsp_tag    (cp_membus_rsp_tag),
        .mem_rsp_ready  (cp_membus_rsp_ready)
    );

    `ASSIGN_VX_MEM_BUS_IF(cp_vx_mem_arb_in_if[0], vx_mem_bus_if[0]);

    assign cp_vx_mem_arb_in_if[1].req_valid       = cp_membus_req_valid;
    assign cp_vx_mem_arb_in_if[1].req_data.rw     = cp_membus_req_rw;
    assign cp_vx_mem_arb_in_if[1].req_data.addr   = cp_membus_req_addr_full[CCI_VX_ADDR_WIDTH-1:0];
    assign cp_vx_mem_arb_in_if[1].req_data.data   = cp_membus_req_data;
    assign cp_vx_mem_arb_in_if[1].req_data.byteen = cp_membus_req_byteen;
    assign cp_vx_mem_arb_in_if[1].req_data.tag    = CCI_VX_TAG_WIDTH'(cp_membus_req_tag);
    assign cp_vx_mem_arb_in_if[1].req_data.attr   = '0;
    assign cp_membus_req_ready                    = cp_vx_mem_arb_in_if[1].req_ready;
    assign cp_membus_rsp_valid = cp_vx_mem_arb_in_if[1].rsp_valid;
    assign cp_membus_rsp_data  = cp_vx_mem_arb_in_if[1].rsp_data.data;
    assign cp_membus_rsp_tag   = cp_vx_mem_arb_in_if[1].rsp_data.tag[`VX_CP_AXI_TID_WIDTH-1:0];
    assign cp_vx_mem_arb_in_if[1].rsp_ready = cp_membus_rsp_ready;

    // High bits of the byte->CL address are unused (CP buffers fit in low
    // device memory) — pin them sink-side so lint stays clean.
    `UNUSED_VAR (cp_membus_req_addr_full[64 - $clog2(LMEM_DATA_WIDTH/8) - 1 : CCI_VX_ADDR_WIDTH])

    VX_mem_bus_if #(
        .DATA_SIZE  (LMEM_DATA_SIZE),
        .ADDR_WIDTH (CCI_VX_ADDR_WIDTH),
        .TAG_WIDTH  (AVS_TAG_WIDTH)
    ) cp_vx_mem_arb_out_if[1]();

    VX_mem_arb #(
        .NUM_INPUTS  (2),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (LMEM_DATA_SIZE),
        .ADDR_WIDTH  (CCI_VX_ADDR_WIDTH),
        .TAG_WIDTH   (CCI_VX_TAG_WIDTH),
        .ARBITER     ("P"), // prioritize Vortex requests; CP shares lower priority
        .REQ_OUT_BUF (0),
        .RSP_OUT_BUF (0)
    ) mem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (cp_vx_mem_arb_in_if),
        .bus_out_if (cp_vx_mem_arb_out_if)
    );
    `UNUSED_VAR (cp_vx_mem_arb_out_if[0].req_data.attr)

    // final merged memory interface
    wire                         mem_req_valid [VX_MEM_PORTS];
    wire                         mem_req_rw [VX_MEM_PORTS];
    wire [CCI_VX_ADDR_WIDTH-1:0] mem_req_addr [VX_MEM_PORTS];
    wire [LMEM_DATA_SIZE-1:0]    mem_req_byteen [VX_MEM_PORTS];
    wire [LMEM_DATA_WIDTH-1:0]   mem_req_data [VX_MEM_PORTS];
    wire [AVS_TAG_WIDTH-1:0]     mem_req_tag [VX_MEM_PORTS];
    wire                         mem_req_ready [VX_MEM_PORTS];

    wire                         mem_rsp_valid [VX_MEM_PORTS];
    wire [LMEM_DATA_WIDTH-1:0]   mem_rsp_data [VX_MEM_PORTS];
    wire [AVS_TAG_WIDTH-1:0]     mem_rsp_tag [VX_MEM_PORTS];
    wire                         mem_rsp_ready [VX_MEM_PORTS];

    for (genvar i = 0; i < VX_MEM_PORTS; ++i) begin : g_mem_bus_if
        if (i == 0) begin : g_i0
            assign mem_req_valid[i] = cp_vx_mem_arb_out_if[i].req_valid;
            assign mem_req_rw[i]    = cp_vx_mem_arb_out_if[i].req_data.rw;
            assign mem_req_addr[i]  = cp_vx_mem_arb_out_if[i].req_data.addr;
            assign mem_req_byteen[i]= cp_vx_mem_arb_out_if[i].req_data.byteen;
            assign mem_req_data[i]  = cp_vx_mem_arb_out_if[i].req_data.data;
            assign mem_req_tag[i]   = cp_vx_mem_arb_out_if[i].req_data.tag;
            assign cp_vx_mem_arb_out_if[i].req_ready = mem_req_ready[i];

            assign cp_vx_mem_arb_out_if[i].rsp_valid     = mem_rsp_valid[i];
            assign cp_vx_mem_arb_out_if[i].rsp_data.data = mem_rsp_data[i];
            assign cp_vx_mem_arb_out_if[i].rsp_data.tag  = mem_rsp_tag[i];
            assign mem_rsp_ready[i] = cp_vx_mem_arb_out_if[i].rsp_ready;
        end else begin : g_i
            assign mem_req_valid[i] = vx_mem_bus_if[i].req_valid;
            assign mem_req_rw[i]    = vx_mem_bus_if[i].req_data.rw;
            assign mem_req_addr[i]  = vx_mem_bus_if[i].req_data.addr;
            assign mem_req_byteen[i]= vx_mem_bus_if[i].req_data.byteen;
            assign mem_req_data[i]  = vx_mem_bus_if[i].req_data.data;
            assign mem_req_tag[i]   = AVS_TAG_WIDTH'(vx_mem_bus_if[i].req_data.tag);
            assign vx_mem_bus_if[i].req_ready = mem_req_ready[i];

            assign vx_mem_bus_if[i].rsp_valid     = mem_rsp_valid[i];
            assign vx_mem_bus_if[i].rsp_data.data = mem_rsp_data[i];
            assign vx_mem_bus_if[i].rsp_data.tag  = CCI_VX_TAG_WIDTH'(mem_rsp_tag[i]);
            assign mem_rsp_ready[i] = vx_mem_bus_if[i].rsp_ready;
        end
    end

    VX_avs_adapter #(
        .DATA_WIDTH    (LMEM_DATA_WIDTH),
        .ADDR_WIDTH_IN (CCI_VX_ADDR_WIDTH),
        .ADDR_WIDTH_OUT(LMEM_ADDR_WIDTH),
        .BURST_WIDTH   (LMEM_BURST_CTRW),
        .NUM_PORTS_IN  (VX_MEM_PORTS),
        .NUM_BANKS_OUT (NUM_LOCAL_MEM_BANKS),
        .TAG_WIDTH     (AVS_TAG_WIDTH),
        .RD_QUEUE_SIZE (AVS_RD_QUEUE_SIZE),
        .INTERLEAVE    (`VX_CFG_PLATFORM_MEMORY_INTERLEAVE),
        .REQ_OUT_BUF   (2),
        .RSP_OUT_BUF   ((VX_MEM_PORTS > 1 || NUM_LOCAL_MEM_BANKS > 1) ? 2 : 0)
    ) avs_adapter (
        .clk              (clk),
        .reset            (reset),

        .mem_req_valid    (mem_req_valid),
        .mem_req_rw       (mem_req_rw),
        .mem_req_byteen   (mem_req_byteen),
        .mem_req_addr     (mem_req_addr),
        .mem_req_data     (mem_req_data),
        .mem_req_tag      (mem_req_tag),
        .mem_req_ready    (mem_req_ready),

        .mem_rsp_valid    (mem_rsp_valid),
        .mem_rsp_data     (mem_rsp_data),
        .mem_rsp_tag      (mem_rsp_tag),
        .mem_rsp_ready    (mem_rsp_ready),

        .avs_writedata    (avs_writedata),
        .avs_readdata     (avs_readdata),
        .avs_address      (avs_address),
        .avs_waitrequest  (avs_waitrequest),
        .avs_write        (avs_write),
        .avs_read         (avs_read),
        .avs_byteenable   (avs_byteenable),
        .avs_burstcount   (avs_burstcount),
        .avs_readdatavalid(avs_readdatavalid)
    );

    // Vortex /////////////////////////////////////////////////////////////////

    reg [VX_DCR_DATA_WIDTH-1:0] dcr_rsp_data_r;
    always @(posedge clk) begin
        if (vx_dcr_rsp_valid) begin
            dcr_rsp_data_r <= vx_dcr_rsp_data;
        end
    end
    `UNUSED_VAR (dcr_rsp_data_r)

    `SCOPE_IO_SWITCH (2);

    Vortex vortex (
        `SCOPE_IO_BIND  (1)

        .clk            (clk),
        .reset          (vx_reset),

        .mem_req_valid  (vx_mem_req_valid),
        .mem_req_rw     (vx_mem_req_rw),
        .mem_req_byteen (vx_mem_req_byteen),
        .mem_req_addr   (vx_mem_req_addr),
        .mem_req_data   (vx_mem_req_data),
        .mem_req_tag    (vx_mem_req_tag),
        .mem_req_ready  (vx_mem_req_ready),

        .mem_rsp_valid  (vx_mem_rsp_valid),
        .mem_rsp_data   (vx_mem_rsp_data),
        .mem_rsp_tag    (vx_mem_rsp_tag),
        .mem_rsp_ready  (vx_mem_rsp_ready),

        .dcr_req_valid  (vx_dcr_req_valid),
        .dcr_req_rw     (vx_dcr_req_rw),
        .dcr_req_addr   (vx_dcr_req_addr),
        .dcr_req_data   (vx_dcr_req_data),
        .dcr_rsp_valid  (vx_dcr_rsp_valid),
        .dcr_rsp_data   (vx_dcr_rsp_data),

        .start          (vx_start),
        .busy           (vx_busy)
    );

    // SCOPE //////////////////////////////////////////////////////////////////

`ifdef DBG_SCOPE_AFU
    wire vx_mem_req_fire = vx_mem_req_valid[0] && vx_mem_req_ready[0];
    wire vx_mem_rsp_fire = vx_mem_rsp_valid[0] && vx_mem_rsp_ready[0];
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP (0, 0, {
            vx_reset,
            vx_busy,
            vx_start,
            vx_mem_req_valid[0],
            vx_mem_req_ready[0],
            vx_mem_rsp_valid[0],
            vx_mem_rsp_ready[0],
            af2cp_sTxPort.c0.valid,
            af2cp_sTxPort.c1.valid,
            cp2af_sRxPort.c0.rspValid,
            cp2af_sRxPort.c1.rspValid,
            cp2af_sRxPort.c0TxAlmFull,
            cp2af_sRxPort.c1TxAlmFull
        },{
            vx_dcr_req_valid,
            cp2af_sRxPort.c0.mmioRdValid,
            cp2af_sRxPort.c0.mmioWrValid,
            af2cp_sTxPort.c2.mmioRdValid,
            vx_mem_req_fire,
            vx_mem_rsp_fire
        },{
            vx_mem_req_rw[0],
            vx_mem_req_addr[0],
            vx_mem_req_tag[0],
            vx_dcr_req_addr,
            vx_dcr_req_data,
            mmio_req_hdr.address,
            af2cp_sTxPort.c0.hdr.address,
            af2cp_sTxPort.c1.hdr.address
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif

`ifdef DBG_TRACE_AFU
    always @(posedge clk) begin
        for (integer i = 0; i < NUM_LOCAL_MEM_BANKS; ++i) begin
            if (avs_write[i] && ~avs_waitrequest[i]) begin
                `TRACE(2, ("%t: AVS Wr Req[%0d]: addr=0x%0h, byteen=0x%0h, burst=0x%0h\n", $time, i, `TO_FULL_ADDR(avs_address[i]), avs_byteenable[i], avs_burstcount[i]))
            end
            if (avs_read[i] && ~avs_waitrequest[i]) begin
                `TRACE(2, ("%t: AVS Rd Req[%0d]: addr=0x%0h, burst=0x%0h\n", $time, i, `TO_FULL_ADDR(avs_address[i]), avs_burstcount[i]))
            end
            if (avs_readdatavalid[i]) begin
                `TRACE(2, ("%t: AVS Rd Rsp[%0d]: data=0x%h\n", $time, i, avs_readdata[i]))
            end
        end
    end
`endif

endmodule
