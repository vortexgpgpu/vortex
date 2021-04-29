`include "VX_platform.vh"
`ifdef NOPAE
`IGNORE_WARNINGS_BEGIN
`include "vortex_afu.vh"
`IGNORE_WARNINGS_END
`else
`include "afu_json_info.vh"
`endif

/* verilator lint_off IMPORTSTAR */ 
import ccip_if_pkg::*;
import local_mem_cfg_pkg::*;
/* verilator lint_on IMPORTSTAR */

`include "VX_define.vh"

module vortex_afu #(
  parameter NUM_LOCAL_MEM_BANKS = 2
) (
  // global signals
  input clk,
  input reset,

  // IF signals between CCI and AFU
  input   t_if_ccip_Rx  cp2af_sRxPort,
  output  t_if_ccip_Tx  af2cp_sTxPort,

  // Avalon signals for local memory access
  output  t_local_mem_data      avs_writedata,
  input   t_local_mem_data      avs_readdata,
  output  t_local_mem_addr      avs_address,
  input   logic                 avs_waitrequest,
  output  logic                 avs_write,
  output  logic                 avs_read,
  output  t_local_mem_byte_mask avs_byteenable,
  output  t_local_mem_burst_cnt avs_burstcount,
  input                         avs_readdatavalid,

  output logic [$clog2(NUM_LOCAL_MEM_BANKS)-1:0] mem_bank_select
);

localparam RESET_DELAY        = 3;  

localparam LMEM_LINE_WIDTH    = $bits(t_local_mem_data);
localparam LMEM_ADDR_WIDTH    = $bits(t_local_mem_addr);
localparam LMEM_BURST_CTRW    = $bits(t_local_mem_burst_cnt);
localparam LMEM_LINE_LW       = $clog2(LMEM_LINE_WIDTH);

localparam CCI_LINE_WIDTH     = $bits(t_ccip_clData);
localparam CCI_ADDR_WIDTH     = 32 - $clog2(CCI_LINE_WIDTH / 8);

localparam VX_MEM_LINE_LW     = $clog2(`VX_MEM_LINE_WIDTH);
localparam VX_MEM_LINE_IDX    = (LMEM_LINE_LW - VX_MEM_LINE_LW);

localparam AVS_RD_QUEUE_SIZE  = 16;
localparam AVS_REQ_TAGW       = `VX_MEM_TAG_WIDTH + VX_MEM_LINE_IDX;

localparam CCI_RD_WINDOW_SIZE = 8;
localparam CCI_RD_QUEUE_SIZE  = 2 * CCI_RD_WINDOW_SIZE;
localparam CCI_RW_PENDING_SIZE= 256;

localparam AFU_ID_L           = 16'h0002;      // AFU ID Lower
localparam AFU_ID_H           = 16'h0004;      // AFU ID Higher 

localparam CMD_MEM_READ       = `AFU_IMAGE_CMD_MEM_READ;
localparam CMD_MEM_WRITE      = `AFU_IMAGE_CMD_MEM_WRITE;
localparam CMD_RUN            = `AFU_IMAGE_CMD_RUN;
localparam CMD_CSR_READ       = `AFU_IMAGE_CMD_CSR_READ;
localparam CMD_CSR_WRITE      = `AFU_IMAGE_CMD_CSR_WRITE;

localparam MMIO_CMD_TYPE      = `AFU_IMAGE_MMIO_CMD_TYPE; 
localparam MMIO_IO_ADDR       = `AFU_IMAGE_MMIO_IO_ADDR;
localparam MMIO_MEM_ADDR      = `AFU_IMAGE_MMIO_MEM_ADDR;
localparam MMIO_DATA_SIZE     = `AFU_IMAGE_MMIO_DATA_SIZE;
localparam MMIO_STATUS        = `AFU_IMAGE_MMIO_STATUS;

localparam MMIO_SCOPE_READ    = `AFU_IMAGE_MMIO_SCOPE_READ;
localparam MMIO_SCOPE_WRITE   = `AFU_IMAGE_MMIO_SCOPE_WRITE;

localparam MMIO_CSR_CORE      = `AFU_IMAGE_MMIO_CSR_CORE;
localparam MMIO_CSR_ADDR      = `AFU_IMAGE_MMIO_CSR_ADDR;
localparam MMIO_CSR_DATA      = `AFU_IMAGE_MMIO_CSR_DATA;
localparam MMIO_CSR_READ      = `AFU_IMAGE_MMIO_CSR_READ;

localparam CCI_RD_RQ_TAGW     = $clog2(CCI_RD_WINDOW_SIZE);
localparam CCI_RD_RQ_DATAW    = CCI_LINE_WIDTH + CCI_RD_RQ_TAGW;

localparam STATE_IDLE         = 0;
localparam STATE_READ         = 1;
localparam STATE_WRITE        = 2;
localparam STATE_START        = 3;
localparam STATE_RUN          = 4;
localparam STATE_CSR_READ     = 5;
localparam STATE_CSR_WRITE    = 6;
localparam STATE_MAX_VALUE    = 7;
localparam STATE_WIDTH        = $clog2(STATE_MAX_VALUE);

`ifdef SCOPE
`SCOPE_DECL_SIGNALS
`endif

wire [127:0] afu_id = `AFU_ACCEL_UUID;

reg [STATE_WIDTH-1:0] state;

// Vortex ports ///////////////////////////////////////////////////////////////

wire vx_mem_req_valid;
wire vx_mem_req_rw;
wire [`VX_MEM_BYTEEN_WIDTH-1:0] vx_mem_req_byteen;
wire [`VX_MEM_ADDR_WIDTH-1:0]   vx_mem_req_addr;
wire [`VX_MEM_LINE_WIDTH-1:0]   vx_mem_req_data;
wire [`VX_MEM_TAG_WIDTH-1:0]    vx_mem_req_tag;
wire vx_mem_req_ready;

wire vx_mem_rsp_valid;
wire [`VX_MEM_LINE_WIDTH-1:0]   vx_mem_rsp_data;
wire [`VX_MEM_TAG_WIDTH-1:0]    vx_mem_rsp_tag;
wire vx_mem_rsp_ready;

wire        vx_csr_io_req_valid;
wire [`VX_CSR_ID_WIDTH-1:0] vx_csr_io_req_coreid;
wire [11:0] vx_csr_io_req_addr;
wire        vx_csr_io_req_rw;
wire [31:0] vx_csr_io_req_data;
wire        vx_csr_io_req_ready;

wire        vx_csr_io_rsp_valid;
wire [31:0] vx_csr_io_rsp_data;
wire        vx_csr_io_rsp_ready;

wire        vx_busy;

reg vx_reset;
reg vx_mem_en;

// CMD variables //////////////////////////////////////////////////////////////

t_ccip_clAddr             cmd_io_addr;
reg [CCI_ADDR_WIDTH-1:0]  cmd_mem_addr;
reg [CCI_ADDR_WIDTH-1:0]  cmd_data_size;

`ifdef SCOPE
wire [63:0]               cmd_scope_rdata;
wire [63:0]               cmd_scope_wdata;  
wire                      cmd_scope_read;
wire                      cmd_scope_write;
`endif

reg [`VX_CSR_ID_WIDTH-1:0] cmd_csr_core;
reg [11:0]                cmd_csr_addr;
reg [31:0]                cmd_csr_rdata;  
reg [31:0]                cmd_csr_wdata;

// MMIO controller ////////////////////////////////////////////////////////////

`IGNORE_WARNINGS_BEGIN
t_ccip_c0_ReqMmioHdr mmio_hdr;
`IGNORE_WARNINGS_END
assign mmio_hdr = t_ccip_c0_ReqMmioHdr'(cp2af_sRxPort.c0.hdr);

`STATIC_ASSERT(($bits(t_ccip_c0_ReqMmioHdr)-$bits(mmio_hdr.address)) == 12, ("Oops!"))

t_if_ccip_c2_Tx mmio_tx;
assign af2cp_sTxPort.c2 = mmio_tx;

`ifdef SCOPE
assign cmd_scope_wdata = 64'(cp2af_sRxPort.c0.data);
assign cmd_scope_read  = cp2af_sRxPort.c0.mmioRdValid && (MMIO_SCOPE_READ == mmio_hdr.address);
assign cmd_scope_write = cp2af_sRxPort.c0.mmioWrValid && (MMIO_SCOPE_WRITE == mmio_hdr.address);
`endif

/*
`DEBUG_BEGIN
wire cp2af_sRxPort_c0_mmioWrValid = cp2af_sRxPort.c0.mmioWrValid;
wire cp2af_sRxPort_c0_mmioRdValid = cp2af_sRxPort.c0.mmioRdValid;
wire cp2af_sRxPort_c0_rspValid = cp2af_sRxPort.c0.rspValid;
wire cp2af_sRxPort_c1_rspValid = cp2af_sRxPort.c1.rspValid;
wire cp2af_sRxPort_c0TxAlmFull = cp2af_sRxPort.c0TxAlmFull;
wire cp2af_sRxPort_c1TxAlmFull = cp2af_sRxPort.c1TxAlmFull;
wire[$bits(mmio_hdr.address)-1:0] mmio_hdr_address = mmio_hdr.address;
wire[$bits(mmio_hdr.length)-1:0] mmio_hdr_length = mmio_hdr.length;
wire[$bits(mmio_hdr.tid)-1:0] mmio_hdr_tid = mmio_hdr.tid;
wire[$bits(cp2af_sRxPort.c0.hdr.mdata)-1:0] cp2af_sRxPort_c0_hdr_mdata = cp2af_sRxPort.c0.hdr.mdata;
`DEBUG_END
*/

wire [2:0] cmd_type = (cp2af_sRxPort.c0.mmioWrValid 
                    && (MMIO_CMD_TYPE == mmio_hdr.address)) ? 3'(cp2af_sRxPort.c0.data) : 3'h0;

// disable assertions until full reset
`ifndef VERILATOR
reg [$clog2(RESET_DELAY+1)-1:0] assert_delay_ctr;
initial begin
  $assertoff;  
end
always @(posedge clk) begin
  if (reset) begin
    assert_delay_ctr <= 0;
  end else begin
    assert_delay_ctr <= assert_delay_ctr + 1;
    if (assert_delay_ctr == RESET_DELAY) begin
      $asserton; // enable assertions
    end
  end
end
`endif

always @(posedge clk) begin
  if (reset) begin
    mmio_tx.mmioRdValid <= 0;
    mmio_tx.hdr         <= 0;
  end else begin
    mmio_tx.mmioRdValid <= cp2af_sRxPort.c0.mmioRdValid; 
    mmio_tx.hdr.tid     <= mmio_hdr.tid;
  end

  // serve MMIO write request
  if (cp2af_sRxPort.c0.mmioWrValid) begin
    case (mmio_hdr.address)
      MMIO_IO_ADDR: begin
        cmd_io_addr <= t_ccip_clAddr'(cp2af_sRxPort.c0.data);
      `ifdef DBG_PRINT_OPAE 
        $display("%t: MMIO_IO_ADDR: addr=%0h, data=0x%0h", $time, mmio_hdr.address, t_ccip_clAddr'(cp2af_sRxPort.c0.data));
      `endif
      end
      MMIO_MEM_ADDR: begin
        cmd_mem_addr <= $bits(cmd_mem_addr)'(cp2af_sRxPort.c0.data);
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_MEM_ADDR: addr=%0h, data=0x%0h", $time, mmio_hdr.address, $bits(cmd_mem_addr)'(cp2af_sRxPort.c0.data));
      `endif
      end
      MMIO_DATA_SIZE: begin
        cmd_data_size <= $bits(cmd_data_size)'(cp2af_sRxPort.c0.data);
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_DATA_SIZE: addr=%0h, data=%0d", $time, mmio_hdr.address, $bits(cmd_data_size)'(cp2af_sRxPort.c0.data));
      `endif
      end
      MMIO_CMD_TYPE: begin
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_CMD_TYPE: addr=%0h, data=%0d", $time, mmio_hdr.address, $bits(cmd_type)'(cp2af_sRxPort.c0.data));
      `endif
      end
    `ifdef SCOPE
      MMIO_SCOPE_WRITE: begin
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_SCOPE_WRITE: addr=%0h, data=%0h", $time, mmio_hdr.address, 64'(cp2af_sRxPort.c0.data));
      `endif
      end
    `endif
      MMIO_CSR_CORE: begin
        cmd_csr_core <= $bits(cmd_csr_core)'(cp2af_sRxPort.c0.data);
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_CSR_CORE: addr=%0h, %0h", $time, mmio_hdr.address, $bits(cmd_csr_core)'(cp2af_sRxPort.c0.data));
      `endif
      end
      MMIO_CSR_ADDR: begin
        cmd_csr_addr <= $bits(cmd_csr_addr)'(cp2af_sRxPort.c0.data);
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_CSR_ADDR: addr=%0h, %0h", $time, mmio_hdr.address, $bits(cmd_csr_addr)'(cp2af_sRxPort.c0.data));
      `endif
      end
      MMIO_CSR_DATA: begin
        cmd_csr_wdata <= $bits(cmd_csr_wdata)'(cp2af_sRxPort.c0.data);
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_CSR_DATA: addr=%0h, %0h", $time, mmio_hdr.address, $bits(cmd_csr_wdata)'(cp2af_sRxPort.c0.data));
      `endif
      end
      default: begin
        `ifdef DBG_PRINT_OPAE
          $display("%t: Unknown MMIO Wr: addr=%0h, data=%0h", $time, mmio_hdr.address, $bits(cmd_csr_wdata)'(cp2af_sRxPort.c0.data));
        `endif
      end
    endcase
  end

  // serve MMIO read requests
  if (cp2af_sRxPort.c0.mmioRdValid) begin
    case (mmio_hdr.address)
      // AFU header
      16'h0000: mmio_tx.data <= {
        4'b0001, // Feature type = AFU
        8'b0,    // reserved
        4'b0,    // afu minor revision = 0
        7'b0,    // reserved
        1'b1,    // end of DFH list = 1 
        24'b0,   // next DFH offset = 0
        4'b0,    // afu major revision = 0
        12'b0    // feature ID = 0
      };
      AFU_ID_L: mmio_tx.data <= afu_id[63:0];   // afu id low
      AFU_ID_H: mmio_tx.data <= afu_id[127:64]; // afu id hi
      16'h0006: mmio_tx.data <= 64'h0; // next AFU
      16'h0008: mmio_tx.data <= 64'h0; // reserved
      MMIO_STATUS: begin
        mmio_tx.data <= 64'(state);
      `ifdef DBG_PRINT_OPAE
        if (state != STATE_WIDTH'(mmio_tx.data)) begin
          $display("%t: MMIO_STATUS: addr=%0h, state=%0d", $time, mmio_hdr.address, state);
        end
      `endif
      end
      MMIO_CSR_READ: begin
        mmio_tx.data <= 64'(cmd_csr_rdata);
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_CSR_READ: addr=%0h, data=%0h", $time, mmio_hdr.address, cmd_csr_rdata);
      `endif
      end
    `ifdef SCOPE
      MMIO_SCOPE_READ: begin
        mmio_tx.data <= cmd_scope_rdata;
      `ifdef DBG_PRINT_OPAE
        $display("%t: MMIO_SCOPE_READ: addr=%0h, data=%0h", $time, mmio_hdr.address, cmd_scope_rdata);
      `endif
      end
    `endif
      default: begin
        mmio_tx.data <= 64'h0;
      `ifdef DBG_PRINT_OPAE
        $display("%t: Unknown MMIO Rd: addr=%0h", $time, mmio_hdr.address);
      `endif
      end
    endcase
  end
end

// COMMAND FSM ////////////////////////////////////////////////////////////////

wire cmd_read_done;
wire cmd_write_done;
wire cmd_csr_done;
wire cmd_run_done;

reg [$clog2(RESET_DELAY+1)-1:0] vx_reset_ctr;
always @(posedge clk) begin
  if (state == STATE_IDLE) begin
    vx_reset_ctr <= 0;
  end else if (state == STATE_START) begin
    vx_reset_ctr <= vx_reset_ctr + 1;
  end
end

always @(posedge clk) begin
  if (reset) begin
    state      <= STATE_IDLE;    
    vx_reset   <= 0;    
    vx_mem_en  <= 0;
  end else begin
    case (state)
      STATE_IDLE: begin             
        case (cmd_type)
          CMD_MEM_READ: begin     
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE READ: ia=%0h addr=%0h size=%0d", $time, cmd_io_addr, cmd_mem_addr, cmd_data_size);
          `endif
            state <= STATE_READ;   
          end 
          CMD_MEM_WRITE: begin      
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE WRITE: ia=%0h addr=%0h size=%0d", $time, cmd_io_addr, cmd_mem_addr, cmd_data_size);
          `endif
            state <= STATE_WRITE;
          end
          CMD_RUN: begin        
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE START", $time);
          `endif
            vx_reset <= 1;            
            state <= STATE_START;                    
          end
          CMD_CSR_READ: begin
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE CSR_READ: addr=%0h", $time, cmd_csr_addr);
          `endif
            state <= STATE_CSR_READ;
          end
          CMD_CSR_WRITE: begin
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE CSR_WRITE: addr=%0h data=%0d", $time, cmd_csr_addr, cmd_csr_wdata);
          `endif
            state <= STATE_CSR_WRITE;
          end
          default: begin
            state <= state;
          end
        endcase
      end      

      STATE_READ: begin
        if (cmd_read_done) begin
          state <= STATE_IDLE;
        `ifdef DBG_PRINT_OPAE
          $display("%t: STATE IDLE", $time);
        `endif
        end
      end

      STATE_WRITE: begin
        if (cmd_write_done) begin
          state <= STATE_IDLE;
        `ifdef DBG_PRINT_OPAE
          $display("%t: STATE IDLE", $time);
        `endif
        end
      end

      STATE_START: begin 
        // vortex reset cycles
        if (vx_reset_ctr == $bits(vx_reset_ctr)'(RESET_DELAY)) begin
          vx_reset   <= 0;  
          vx_mem_en  <= 1;
          state <= STATE_RUN;
        end
      end

      STATE_RUN: begin
        if (cmd_run_done) begin
          vx_mem_en <= 0;
          state <= STATE_IDLE;
        `ifdef DBG_PRINT_OPAE
          $display("%t: STATE IDLE", $time);
        `endif
        end
      end

      STATE_CSR_READ: begin
        if (cmd_csr_done) begin
          state <= STATE_IDLE;
        `ifdef DBG_PRINT_OPAE
          $display("%t: STATE IDLE", $time);
        `endif
        end
      end

      STATE_CSR_WRITE: begin
        if (cmd_csr_done) begin
          state <= STATE_IDLE;
        `ifdef DBG_PRINT_OPAE
          $display("%t: STATE IDLE", $time);
        `endif
        end
      end

      default: begin
        state <= state;
      end

    endcase
  end
end

// AVS Controller /////////////////////////////////////////////////////////////

wire                    mem_req_valid;
wire                    mem_req_rw; 
t_local_mem_byte_mask   mem_req_byteen;
t_local_mem_addr        mem_req_addr;
t_local_mem_data        mem_req_data;
wire [AVS_REQ_TAGW:0]   mem_req_tag;
wire                    mem_req_ready;

wire                    mem_rsp_valid;        
t_local_mem_data        mem_rsp_data;
wire [AVS_REQ_TAGW:0]   mem_rsp_tag;
wire                    mem_rsp_ready;

wire                    cci_mem_req_tmp_valid;
wire                    cci_mem_req_tmp_rw; 
t_local_mem_byte_mask   cci_mem_req_tmp_byteen;
t_local_mem_addr        cci_mem_req_tmp_addr;
t_local_mem_data        cci_mem_req_tmp_data;
wire [AVS_REQ_TAGW-1:0] cci_mem_req_tmp_tag;
wire                    cci_mem_req_tmp_ready;

wire                    cci_mem_rsp_tmp_valid;        
t_local_mem_data        cci_mem_rsp_tmp_data;
wire [AVS_REQ_TAGW-1:0] cci_mem_rsp_tmp_tag;
wire                    cci_mem_rsp_tmp_ready;

wire                    vx_mem_req_valid_qual;
t_local_mem_addr        vx_mem_req_addr_qual;
t_local_mem_byte_mask   vx_mem_req_byteen_qual;
t_local_mem_data        vx_mem_req_data_qual;
wire [AVS_REQ_TAGW-1:0] vx_mem_req_tag_qual;

wire [(1 << VX_MEM_LINE_IDX)-1:0][`VX_MEM_LINE_WIDTH-1:0] vx_mem_rsp_data_unqual;
wire [AVS_REQ_TAGW-1:0] vx_mem_rsp_tag_unqual;

wire cci_mem_rd_req_valid;
wire cci_mem_wr_req_valid;
wire [CCI_ADDR_WIDTH-1:0] cci_mem_rd_req_addr;
wire [CCI_ADDR_WIDTH-1:0] cci_mem_wr_req_addr;
wire [CCI_RD_RQ_DATAW-1:0] cci_rdq_dout;
wire cci_mem_req_ready;

wire cci_mem_rsp_valid;        
wire [CCI_LINE_WIDTH-1:0] cci_mem_rsp_data;
wire [AVS_REQ_TAGW-1:0] cci_mem_rsp_tag;
wire cci_mem_rsp_ready;

//--

VX_cci_to_mem #(
  .CCI_DATAW (CCI_LINE_WIDTH), 
  .CCI_ADDRW (CCI_ADDR_WIDTH),           
  .AVS_DATAW (LMEM_LINE_WIDTH), 
  .AVS_ADDRW (LMEM_ADDR_WIDTH),         
  .TAG_WIDTH (AVS_REQ_TAGW)
) cci_to_mem(
  .clk    (clk),
  .reset  (reset),

  .mem_req_valid_in  ((CMD_MEM_WRITE == state) ? cci_mem_wr_req_valid : cci_mem_rd_req_valid),
  .mem_req_addr_in   ((CMD_MEM_WRITE == state) ? cci_mem_wr_req_addr : cci_mem_rd_req_addr),
  .mem_req_rw_in     ((CMD_MEM_WRITE == state)),
  .mem_req_data_in   (cci_rdq_dout[CCI_RD_RQ_DATAW-1:CCI_RD_RQ_TAGW]),
  .mem_req_tag_in    (AVS_REQ_TAGW'(0)), 
  .mem_req_ready_in  (cci_mem_req_ready), 

  .mem_req_valid_out (cci_mem_req_tmp_valid),
  .mem_req_addr_out  (cci_mem_req_tmp_addr),
  .mem_req_rw_out    (cci_mem_req_tmp_rw),
  .mem_req_byteen_out(cci_mem_req_tmp_byteen),
  .mem_req_data_out  (cci_mem_req_tmp_data),
  .mem_req_tag_out   (cci_mem_req_tmp_tag),
  .mem_req_ready_out (cci_mem_req_tmp_ready), 

  .mem_rsp_valid_in  (cci_mem_rsp_tmp_valid), 
  .mem_rsp_data_in   (cci_mem_rsp_tmp_data), 
  .mem_rsp_tag_in    (cci_mem_rsp_tmp_tag), 
  .mem_rsp_ready_in  (cci_mem_rsp_tmp_ready),

  .mem_rsp_valid_out (cci_mem_rsp_valid), 
  .mem_rsp_data_out  (cci_mem_rsp_data), 
  .mem_rsp_tag_out   (cci_mem_rsp_tag), 
  .mem_rsp_ready_out (cci_mem_rsp_ready) 
);

`UNUSED_VAR (cci_mem_rsp_tag)

//--

assign vx_mem_req_valid_qual = vx_mem_req_valid && vx_mem_en;

assign vx_mem_req_addr_qual = vx_mem_req_addr[`VX_MEM_ADDR_WIDTH-1:`VX_MEM_ADDR_WIDTH-LMEM_ADDR_WIDTH];

if (`VX_MEM_LINE_WIDTH != LMEM_LINE_WIDTH) begin
  wire [VX_MEM_LINE_IDX-1:0] vx_mem_req_idx = vx_mem_req_addr[VX_MEM_LINE_IDX-1:0];
  wire [VX_MEM_LINE_IDX-1:0] vx_mem_rsp_idx = vx_mem_rsp_tag_unqual[VX_MEM_LINE_IDX-1:0];
  assign vx_mem_req_byteen_qual = 64'(vx_mem_req_byteen) << (6'(vx_mem_req_addr[VX_MEM_LINE_IDX-1:0]) << (VX_MEM_LINE_LW-3));  
  assign vx_mem_req_data_qual   = LMEM_LINE_WIDTH'(vx_mem_req_data) << ((LMEM_LINE_LW'(vx_mem_req_idx)) << VX_MEM_LINE_LW);
  assign vx_mem_req_tag_qual    = {vx_mem_req_tag, vx_mem_req_idx};
  assign vx_mem_rsp_data        = vx_mem_rsp_data_unqual[vx_mem_rsp_idx];  
end else begin
  assign vx_mem_req_byteen_qual = vx_mem_req_byteen;
  assign vx_mem_req_tag_qual    = vx_mem_req_tag;
  assign vx_mem_req_data_qual   = vx_mem_req_data;
  assign vx_mem_rsp_data        = vx_mem_rsp_data_unqual;
end

assign vx_mem_rsp_tag = vx_mem_rsp_tag_unqual[`VX_MEM_TAG_WIDTH+VX_MEM_LINE_IDX-1:VX_MEM_LINE_IDX];

//--

VX_mem_arb #(
  .NUM_REQS      (2),
  .DATA_WIDTH    (LMEM_LINE_WIDTH),
  .ADDR_WIDTH    (LMEM_ADDR_WIDTH),
  .TAG_IN_WIDTH  (AVS_REQ_TAGW),
  .TAG_OUT_WIDTH (AVS_REQ_TAGW+1)
) mem_arb (
  .clk            (clk),
  .reset          (reset),

  // Source request
  .req_valid_in   ({cci_mem_req_tmp_valid,  vx_mem_req_valid_qual}),
  .req_rw_in      ({cci_mem_req_tmp_rw,     vx_mem_req_rw}),
  .req_byteen_in  ({cci_mem_req_tmp_byteen, vx_mem_req_byteen_qual}),
  .req_addr_in    ({cci_mem_req_tmp_addr,   vx_mem_req_addr_qual}),
  .req_data_in    ({cci_mem_req_tmp_data,   vx_mem_req_data_qual}),  
  .req_tag_in     ({cci_mem_req_tmp_tag,    vx_mem_req_tag_qual}),  
  .req_ready_in   ({cci_mem_req_tmp_ready,  vx_mem_req_ready}),

  // Memory request
  .req_valid_out  (mem_req_valid),
  .req_rw_out     (mem_req_rw),        
  .req_byteen_out (mem_req_byteen),        
  .req_addr_out   (mem_req_addr),
  .req_data_out   (mem_req_data),
  .req_tag_out    (mem_req_tag),
  .req_ready_out  (mem_req_ready),

  // Source response
  .rsp_valid_out  ({cci_mem_rsp_tmp_valid, vx_mem_rsp_valid}),
  .rsp_data_out   ({cci_mem_rsp_tmp_data,  vx_mem_rsp_data_unqual}),
  .rsp_tag_out    ({cci_mem_rsp_tmp_tag,   vx_mem_rsp_tag_unqual}),
  .rsp_ready_out  ({cci_mem_rsp_tmp_ready, vx_mem_rsp_ready}),
  
  // Memory response
  .rsp_valid_in   (mem_rsp_valid),
  .rsp_tag_in     (mem_rsp_tag),
  .rsp_data_in    (mem_rsp_data),
  .rsp_ready_in   (mem_rsp_ready)
);

//--

VX_avs_wrapper #(
  .AVS_DATAW     (LMEM_LINE_WIDTH), 
  .AVS_ADDRW     (LMEM_ADDR_WIDTH),
  .AVS_BURSTW    (LMEM_BURST_CTRW),
  .AVS_BANKS     (NUM_LOCAL_MEM_BANKS),
  .REQ_TAGW      (AVS_REQ_TAGW+1),
  .RD_QUEUE_SIZE (AVS_RD_QUEUE_SIZE)
) avs_wrapper (
  .clk                (clk),
  .reset              (reset),

  // Memory request 
  .mem_req_valid     (mem_req_valid),
  .mem_req_rw        (mem_req_rw),
  .mem_req_byteen    (mem_req_byteen),
  .mem_req_addr      (mem_req_addr),
  .mem_req_data      (mem_req_data),
  .mem_req_tag       (mem_req_tag),
  .mem_req_ready     (mem_req_ready),

  // Memory response  
  .mem_rsp_valid     (mem_rsp_valid),
  .mem_rsp_data      (mem_rsp_data),
  .mem_rsp_tag       (mem_rsp_tag),
  .mem_rsp_ready     (mem_rsp_ready),

  // AVS bus
  .avs_writedata      (avs_writedata),
  .avs_readdata       (avs_readdata),
  .avs_address        (avs_address),
  .avs_waitrequest    (avs_waitrequest),
  .avs_write          (avs_write),
  .avs_read           (avs_read),
  .avs_byteenable     (avs_byteenable),
  .avs_burstcount     (avs_burstcount),
  .avs_readdatavalid  (avs_readdatavalid),
  .avs_bankselect     (mem_bank_select)
);

// CCI-P Read Request ///////////////////////////////////////////////////////////

reg [CCI_ADDR_WIDTH-1:0] cci_mem_wr_req_ctr;
reg [CCI_ADDR_WIDTH-1:0] cci_rd_req_ctr;
wire [CCI_ADDR_WIDTH-1:0] cci_rd_req_ctr_next;
reg [CCI_ADDR_WIDTH-1:0] cci_mem_wr_req_addr_unqual;
wire [CCI_RD_RQ_TAGW-1:0] cci_rd_req_tag, cci_rd_rsp_tag;
reg [CCI_RD_RQ_TAGW-1:0] cci_rd_rsp_ctr;
t_ccip_clAddr cci_rd_req_addr;

reg cci_rd_req_enable, cci_rd_req_wait;

wire cci_rdq_push, cci_rdq_pop;
wire [CCI_RD_RQ_DATAW-1:0] cci_rdq_din;
wire cci_rdq_empty;

always @(*) begin
  af2cp_sTxPort.c0.hdr         = t_ccip_c0_ReqMemHdr'(0);
  af2cp_sTxPort.c0.hdr.address = cci_rd_req_addr;  
  af2cp_sTxPort.c0.hdr.mdata   = t_ccip_mdata'(cci_rd_req_tag);
end

wire cci_mem_wr_req_fire = cci_mem_wr_req_valid && cci_mem_req_ready;

wire cci_rd_req_fire = af2cp_sTxPort.c0.valid;

wire cci_rd_rsp_fire = (STATE_WRITE == state) 
                    && cp2af_sRxPort.c0.rspValid 
                    && (cp2af_sRxPort.c0.hdr.resp_type == eRSP_RDLINE);

assign cci_rd_req_tag = CCI_RD_RQ_TAGW'(cci_rd_req_ctr);
assign cci_rd_rsp_tag = CCI_RD_RQ_TAGW'(cp2af_sRxPort.c0.hdr.mdata);

assign cci_rd_req_ctr_next = cci_rd_req_ctr + CCI_ADDR_WIDTH'(cci_rd_req_fire ? 1 : 0);

assign cci_rdq_pop  = cci_mem_wr_req_fire;
assign cci_rdq_push = cci_rd_rsp_fire;
assign cci_rdq_din  = {cp2af_sRxPort.c0.data, cci_rd_rsp_tag};

wire [$clog2(CCI_RD_QUEUE_SIZE+1)-1:0] cci_pending_reads;
wire cci_pending_reads_full;
VX_pending_size #( 
    .SIZE (CCI_RD_QUEUE_SIZE)
) cci_rd_pending_size (
    .clk   (clk),
    .reset (reset),
    .push  (cci_rd_req_fire),
    .pop   (cci_rdq_pop),
    `UNUSED_PIN (empty),
    .full  (cci_pending_reads_full),
    .size  (cci_pending_reads)
);
`UNUSED_VAR (cci_pending_reads)

assign cci_mem_wr_req_valid = !cci_rdq_empty;

assign cci_mem_wr_req_addr = cci_mem_wr_req_addr_unqual + (CCI_ADDR_WIDTH'(CCI_RD_RQ_TAGW'(cci_rdq_dout)));
                            
assign af2cp_sTxPort.c0.valid = cci_rd_req_enable && !(cci_rd_req_wait || cci_pending_reads_full);

assign cmd_write_done = (cci_mem_wr_req_ctr == cmd_data_size);

// Send read requests to CCI
always @(posedge clk) begin
  if (reset) begin
    cci_rd_req_addr     <= 0;
    cci_rd_req_ctr      <= 0;
    cci_rd_rsp_ctr      <= 0;
    cci_rd_req_enable   <= 0;
    cci_rd_req_wait     <= 0;
    cci_mem_wr_req_ctr <= 0;
    cci_mem_wr_req_addr_unqual <= 0;
  end 
  else begin          
    if ((STATE_IDLE == state) 
    &&  (CMD_MEM_WRITE == cmd_type)) begin
      cci_rd_req_addr    <= cmd_io_addr;
      cci_rd_req_ctr     <= 0;
      cci_rd_rsp_ctr     <= 0;
      cci_rd_req_enable  <= (cmd_data_size != 0);
      cci_rd_req_wait    <= 0;      
      cci_mem_wr_req_ctr <= 0;
      cci_mem_wr_req_addr_unqual <= cmd_mem_addr;
    end

    cci_rd_req_enable <= (STATE_WRITE == state)                       
                      && (cci_rd_req_ctr_next != cmd_data_size)
                      && !cp2af_sRxPort.c0TxAlmFull;    

    if (cci_rd_req_fire) begin  
      cci_rd_req_addr <= cci_rd_req_addr + 1;
      cci_rd_req_ctr  <= cci_rd_req_ctr_next;
      if (cci_rd_req_tag == CCI_RD_RQ_TAGW'(CCI_RD_WINDOW_SIZE-1)) begin
        cci_rd_req_wait <= 1; // end current request batch
      end 
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Req: addr=%0h, tag=%0h, rem=%0d, pending=%0d", $time, cci_rd_req_addr, cci_rd_req_tag, (cmd_data_size - cci_rd_req_ctr_next), cci_pending_reads);
    `endif
    end

    if (cci_rd_rsp_fire) begin
      cci_rd_rsp_ctr <= cci_rd_rsp_ctr + CCI_RD_RQ_TAGW'(1);
      if (cci_rd_rsp_ctr == CCI_RD_RQ_TAGW'(CCI_RD_WINDOW_SIZE-1)) begin
        cci_rd_req_wait <= 0; // restart new request batch
      end 
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Rsp: idx=%0d, ctr=%0d, data=%0h", $time, cci_rd_rsp_tag, cci_rd_rsp_ctr, cp2af_sRxPort.c0.data);
    `endif
    end

    /*if (cci_rdq_pop) begin
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Queue Pop: pending=%0d", $time, cci_pending_reads);
    `endif
    end*/

    if (cci_mem_wr_req_fire) begin                
      cci_mem_wr_req_addr_unqual <= cci_mem_wr_req_addr_unqual + ((CCI_RD_RQ_TAGW'(cci_mem_wr_req_ctr) == CCI_RD_RQ_TAGW'(CCI_RD_WINDOW_SIZE-1)) ? CCI_ADDR_WIDTH'(CCI_RD_WINDOW_SIZE) : CCI_ADDR_WIDTH'(0));
      cci_mem_wr_req_ctr         <= cci_mem_wr_req_ctr + CCI_ADDR_WIDTH'(1);
    end
  end
end

VX_fifo_queue #(
  .DATAW   (CCI_RD_RQ_DATAW),
  .SIZE    (CCI_RD_QUEUE_SIZE)
) cci_rd_req_queue (
  .clk      (clk),
  .reset    (reset),
  .push     (cci_rdq_push),
  .pop      (cci_rdq_pop),
  .data_in  (cci_rdq_din),
  .data_out (cci_rdq_dout),
  .empty    (cci_rdq_empty),
  `UNUSED_PIN (full),
  `UNUSED_PIN (alm_empty),
  `UNUSED_PIN (alm_full),
  `UNUSED_PIN (size)
);

`ifdef VERILATOR
`DEBUG_BLOCK(
  reg [CCI_RD_WINDOW_SIZE-1:0] dbg_cci_rd_rsp_mask;
  always @(posedge clk) begin
    if (reset) begin
      dbg_cci_rd_rsp_mask <= 0;
    end else begin
      if (cci_rd_rsp_fire) begin      
        if (cci_rd_rsp_ctr == 0) begin
          dbg_cci_rd_rsp_mask <= (CCI_RD_WINDOW_SIZE'(1) << cci_rd_rsp_tag);               
        end else begin                
          assert(!dbg_cci_rd_rsp_mask[cci_rd_rsp_tag]);
          dbg_cci_rd_rsp_mask[cci_rd_rsp_tag] <= 1;
        end 
      end
    end
  end
)
`endif

// CCI-P Write Request //////////////////////////////////////////////////////////

reg [CCI_ADDR_WIDTH-1:0] cci_mem_rd_req_ctr;
reg [CCI_ADDR_WIDTH-1:0] cci_wr_req_ctr;
reg [CCI_ADDR_WIDTH-1:0] cci_mem_rd_req_addr_r;
t_ccip_clAddr cci_wr_req_addr;

always @(*) begin
  af2cp_sTxPort.c1.hdr         = t_ccip_c1_ReqMemHdr'(0);
  af2cp_sTxPort.c1.hdr.address = cci_wr_req_addr;
  af2cp_sTxPort.c1.hdr.sop     = 1; // single line write mode
  af2cp_sTxPort.c1.data        = t_ccip_clData'(cci_mem_rsp_data);  
end 

wire cci_mem_rd_req_fire = cci_mem_rd_req_valid && cci_mem_req_ready;
wire cci_mem_rd_rsp_fire = cci_mem_rsp_valid && cci_mem_rsp_ready;

wire cci_wr_req_fire = cci_mem_rd_rsp_fire;

wire cci_wr_rsp_fire = (STATE_READ == state) 
                    && cp2af_sRxPort.c1.rspValid 
                    && (cp2af_sRxPort.c1.hdr.resp_type == eRSP_WRLINE);

wire [$clog2(CCI_RW_PENDING_SIZE+1)-1:0] cci_pending_writes;
wire cci_pending_writes_empty;
wire cci_pending_writes_full;
VX_pending_size #( 
    .SIZE (CCI_RW_PENDING_SIZE)
) cci_wr_pending_size (
    .clk   (clk),
    .reset (reset),
    .push  (cci_wr_req_fire),
    .pop   (cci_wr_rsp_fire),
    .empty (cci_pending_writes_empty),
    .full  (cci_pending_writes_full),
    .size  (cci_pending_writes)
);
`UNUSED_VAR (cci_pending_writes)

assign cci_mem_rd_req_valid = (cci_mem_rd_req_ctr != 0);
assign cci_mem_rd_req_addr  = cci_mem_rd_req_addr_r;

assign af2cp_sTxPort.c1.valid = cci_mem_rd_rsp_fire;
assign cci_mem_rsp_ready = !cp2af_sRxPort.c1TxAlmFull && !cci_pending_writes_full;

assign cmd_read_done = (0 == cci_wr_req_ctr) && cci_pending_writes_empty;

// Send write requests to CCI
always @(posedge clk) 
begin
  if (reset) begin
    cci_wr_req_addr       <= 0;
    cci_wr_req_ctr        <= 0;
    cci_mem_rd_req_ctr    <= 0;
    cci_mem_rd_req_addr_r <= 0;
  end
  else begin    
    if ((STATE_IDLE == state) 
    &&  (CMD_MEM_READ == cmd_type)) begin
      cci_wr_req_addr       <= cmd_io_addr;
      cci_wr_req_ctr        <= cmd_data_size;
      cci_mem_rd_req_ctr    <= cmd_data_size;
      cci_mem_rd_req_addr_r <= cmd_mem_addr;
    end 

    if (cci_wr_req_fire) begin
      assert(cci_wr_req_ctr != 0);  
      cci_wr_req_addr <= cci_wr_req_addr + t_ccip_clAddr'(1);        
      cci_wr_req_ctr  <= cci_wr_req_ctr - CCI_ADDR_WIDTH'(1);
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Wr Req: addr=%0h, rem=%0d, pending=%0d, data=%0h", $time, cci_wr_req_addr, (cci_wr_req_ctr - 1), cci_pending_writes, af2cp_sTxPort.c1.data);
    `endif
    end

  /*`ifdef DBG_PRINT_OPAE
    if (cci_wr_rsp_fire) begin      
      $display("%t: CCI Wr Rsp: pending=%0d", $time, cci_pending_writes);      
    end
  `endif*/

    if (cci_mem_rd_req_fire) begin
      cci_mem_rd_req_addr_r <= cci_mem_rd_req_addr_r + CCI_ADDR_WIDTH'(1);       
      cci_mem_rd_req_ctr    <= cci_mem_rd_req_ctr - CCI_ADDR_WIDTH'(1);
    end
  end
end

// CSRs ///////////////////////////////////////////////////////////////////////

reg csr_io_req_sent;

assign vx_csr_io_req_valid = !csr_io_req_sent 
                          && ((STATE_CSR_READ == state || STATE_CSR_WRITE == state));
assign vx_csr_io_req_coreid = cmd_csr_core;
assign vx_csr_io_req_rw   = (STATE_CSR_WRITE == state);
assign vx_csr_io_req_addr = cmd_csr_addr;
assign vx_csr_io_req_data = cmd_csr_wdata;

assign vx_csr_io_rsp_ready = 1;

assign cmd_csr_done = (STATE_CSR_WRITE == state) ? vx_csr_io_req_ready : vx_csr_io_rsp_valid;

always @(posedge clk) begin
  if (reset) begin
    csr_io_req_sent <= 0;
  end else begin
    if (vx_csr_io_req_valid && vx_csr_io_req_ready) begin
      csr_io_req_sent <= 1;
    end
    if (cmd_csr_done) begin
      csr_io_req_sent <= 0;
    end
  end

  if ((STATE_CSR_READ == state)
   && vx_csr_io_rsp_ready
   && vx_csr_io_rsp_valid) begin
    cmd_csr_rdata <= vx_csr_io_rsp_data;
  end
end

// Vortex /////////////////////////////////////////////////////////////////////

assign cmd_run_done = !vx_busy;

Vortex #() vortex (
  `SCOPE_BIND_afu_vortex

  .clk            (clk),
  .reset          (reset | vx_reset),

  // Memory request 
  .mem_req_valid  (vx_mem_req_valid),
  .mem_req_rw     (vx_mem_req_rw),
  .mem_req_byteen (vx_mem_req_byteen),
  .mem_req_addr   (vx_mem_req_addr),
  .mem_req_data   (vx_mem_req_data),
  .mem_req_tag    (vx_mem_req_tag),
  .mem_req_ready  (vx_mem_req_ready),

  // Memory response  
  .mem_rsp_valid  (vx_mem_rsp_valid),
  .mem_rsp_data   (vx_mem_rsp_data),
  .mem_rsp_tag    (vx_mem_rsp_tag),
  .mem_rsp_ready  (vx_mem_rsp_ready),

  // CSR Request
  .csr_req_valid  (vx_csr_io_req_valid),
  .csr_req_coreid (vx_csr_io_req_coreid),
  .csr_req_addr   (vx_csr_io_req_addr),
  .csr_req_rw     (vx_csr_io_req_rw),
  .csr_req_data   (vx_csr_io_req_data),
  .csr_req_ready  (vx_csr_io_req_ready),

  // CSR Response
  .csr_rsp_valid  (vx_csr_io_rsp_valid),
  .csr_rsp_data   (vx_csr_io_rsp_data),
  .csr_rsp_ready  (vx_csr_io_rsp_ready),
 
  // status
  .busy           (vx_busy),
  `UNUSED_PIN     (ebreak)
);

// SCOPE //////////////////////////////////////////////////////////////////////

`ifdef SCOPE

`SCOPE_ASSIGN (cmd_type, cmd_type);
`SCOPE_ASSIGN (state, state);
`SCOPE_ASSIGN (cci_sRxPort_c0_mmioRdValid, cp2af_sRxPort.c0.mmioRdValid);
`SCOPE_ASSIGN (cci_sRxPort_c0_mmioWrValid, cp2af_sRxPort.c0.mmioWrValid);
`SCOPE_ASSIGN (mmio_hdr_address, mmio_hdr.address);
`SCOPE_ASSIGN (mmio_hdr_length, mmio_hdr.length);
`SCOPE_ASSIGN (cci_sRxPort_c0_hdr_mdata, cp2af_sRxPort.c0.hdr.mdata);
`SCOPE_ASSIGN (cci_sRxPort_c0_rspValid, cp2af_sRxPort.c0.rspValid);
`SCOPE_ASSIGN (cci_sRxPort_c1_rspValid, cp2af_sRxPort.c1.rspValid);            
`SCOPE_ASSIGN (cci_sTxPort_c0_valid, af2cp_sTxPort.c0.valid);
`SCOPE_ASSIGN (cci_sTxPort_c0_hdr_address, af2cp_sTxPort.c0.hdr.address);
`SCOPE_ASSIGN (cci_sTxPort_c0_hdr_mdata, af2cp_sTxPort.c0.hdr.mdata);
`SCOPE_ASSIGN (cci_sTxPort_c1_valid, af2cp_sTxPort.c1.valid);
`SCOPE_ASSIGN (cci_sTxPort_c1_hdr_address, af2cp_sTxPort.c1.hdr.address);
`SCOPE_ASSIGN (cci_sTxPort_c2_mmioRdValid, af2cp_sTxPort.c2.mmioRdValid);
`SCOPE_ASSIGN (cci_sRxPort_c0TxAlmFull, cp2af_sRxPort.c0TxAlmFull);
`SCOPE_ASSIGN (cci_sRxPort_c1TxAlmFull, cp2af_sRxPort.c1TxAlmFull);
`SCOPE_ASSIGN (avs_address, avs_address);
`SCOPE_ASSIGN (avs_waitrequest, avs_waitrequest);
`SCOPE_ASSIGN (avs_write_fire, avs_write && !avs_waitrequest);
`SCOPE_ASSIGN (avs_read_fire, avs_read && !avs_waitrequest);
`SCOPE_ASSIGN (avs_byteenable, avs_byteenable);
`SCOPE_ASSIGN (avs_burstcount, avs_burstcount);
`SCOPE_ASSIGN (avs_readdatavalid, avs_readdatavalid);
`SCOPE_ASSIGN (mem_bank_select, mem_bank_select);          
`SCOPE_ASSIGN (cci_mem_rd_req_ctr, cci_mem_rd_req_ctr);
`SCOPE_ASSIGN (cci_mem_wr_req_ctr, cci_mem_wr_req_ctr);
`SCOPE_ASSIGN (cci_rd_req_ctr, cci_rd_req_ctr);
`SCOPE_ASSIGN (cci_rd_rsp_ctr, cci_rd_rsp_ctr);
`SCOPE_ASSIGN (cci_wr_req_ctr, cci_wr_req_ctr);
`SCOPE_ASSIGN (cci_wr_req_fire, cci_wr_req_fire);
`SCOPE_ASSIGN (cci_wr_rsp_fire, cci_wr_rsp_fire);
`SCOPE_ASSIGN (cci_rd_req_fire, cci_rd_req_fire);
`SCOPE_ASSIGN (cci_rd_rsp_fire, cci_rd_rsp_fire);
`SCOPE_ASSIGN (cci_pending_reads_full, cci_pending_reads_full);
`SCOPE_ASSIGN (cci_pending_writes_empty, cci_pending_writes_empty);
`SCOPE_ASSIGN (cci_pending_writes_full, cci_pending_writes_full);
`SCOPE_ASSIGN (afu_mem_req_fire, (mem_req_valid && mem_req_ready));
`SCOPE_ASSIGN (afu_mem_req_addr, mem_req_addr);
`SCOPE_ASSIGN (afu_mem_req_tag, mem_req_tag);
`SCOPE_ASSIGN (afu_mem_rsp_fire, (mem_rsp_valid && mem_rsp_ready));
`SCOPE_ASSIGN (afu_mem_rsp_tag, mem_rsp_tag);

wire scope_changed = `SCOPE_TRIGGER;

VX_scope #(
  .DATAW ($bits({`SCOPE_DATA_LIST,`SCOPE_UPDATE_LIST})),
  .BUSW  (64),
  .SIZE  (`SCOPE_SIZE),
  .UPDW  ($bits({`SCOPE_UPDATE_LIST}))
) scope (
  .clk      (clk),
  .reset    (reset),
  .start    (1'b0),
  .stop     (1'b0),
  .changed  (scope_changed),
  .data_in  ({`SCOPE_DATA_LIST,`SCOPE_UPDATE_LIST}),
  .bus_in   (cmd_scope_wdata),
  .bus_out  (cmd_scope_rdata),
  .bus_read (cmd_scope_read),
  .bus_write(cmd_scope_write)
);

`else
  `UNUSED_PARAM (MMIO_SCOPE_READ)
  `UNUSED_PARAM (MMIO_SCOPE_WRITE)
`endif

endmodule