`ifndef NOPAE
`include "platform_if.vh"
import local_mem_cfg_pkg::*;
`include "afu_json_info.vh"
`include "VX_define.vh"
`else
`include "vortex_afu.vh"
/* verilator lint_off IMPORTSTAR */ 
import ccip_if_pkg::*;
import local_mem_cfg_pkg::*;
/* verilator lint_on IMPORTSTAR */ 
`endif

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

localparam DRAM_ADDR_WIDTH    = $bits(t_local_mem_addr);
localparam DRAM_LINE_WIDTH    = $bits(t_local_mem_data);

localparam DRAM_LINE_LW       = $clog2(DRAM_LINE_WIDTH);
localparam VX_DRAM_LINE_LW    = $clog2(`VX_DRAM_LINE_WIDTH);

localparam AVS_RD_QUEUE_SIZE  = 16;

localparam CCI_RD_WINDOW_SIZE = 8;
localparam CCI_RD_QUEUE_SIZE  = 2 * CCI_RD_WINDOW_SIZE;
localparam CCI_RW_QUEUE_SIZE  = 1024;

localparam AFU_ID_L           = 16'h0002;      // AFU ID Lower
localparam AFU_ID_H           = 16'h0004;      // AFU ID Higher 

localparam CMD_MEM_READ       = `AFU_IMAGE_CMD_MEM_READ;
localparam CMD_MEM_WRITE      = `AFU_IMAGE_CMD_MEM_WRITE;
localparam CMD_RUN            = `AFU_IMAGE_CMD_RUN;
localparam CMD_CLFLUSH        = `AFU_IMAGE_CMD_CLFLUSH;
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
localparam CCI_RD_RQ_DATAW    = $bits(t_ccip_clData) + CCI_RD_RQ_TAGW;

localparam STATE_IDLE         = 0;
localparam STATE_READ         = 1;
localparam STATE_WRITE        = 2;
localparam STATE_START        = 3;
localparam STATE_RUN          = 4;
localparam STATE_CLFLUSH      = 5;
localparam STATE_CSR_READ     = 6;
localparam STATE_CSR_WRITE    = 7;
localparam STATE_MAX_VALUE    = 8;
localparam STATE_WIDTH        = $clog2(STATE_MAX_VALUE);

`ifdef SCOPE
`SCOPE_DECL_SIGNALS
`endif

wire [127:0] afu_id = `AFU_ACCEL_UUID;

reg [STATE_WIDTH-1:0] state;

// Vortex ports ///////////////////////////////////////////////////////////////

wire vx_dram_req_valid;
wire vx_dram_req_rw;
wire [`VX_DRAM_BYTEEN_WIDTH-1:0] vx_dram_req_byteen;
wire [`VX_DRAM_ADDR_WIDTH-1:0] vx_dram_req_addr;
wire [`VX_DRAM_LINE_WIDTH-1:0] vx_dram_req_data;
wire [`VX_DRAM_TAG_WIDTH-1:0]  vx_dram_req_tag;
wire vx_dram_req_ready;

wire vx_dram_rsp_valid;
wire [`VX_DRAM_LINE_WIDTH-1:0] vx_dram_rsp_data;
wire [`VX_DRAM_TAG_WIDTH-1:0]  vx_dram_rsp_tag;
wire vx_dram_rsp_ready;

reg vx_snp_req_valid;
reg [`VX_DRAM_ADDR_WIDTH-1:0] vx_snp_req_addr;
wire vx_snp_req_invalidate = 0;
reg [`VX_SNP_TAG_WIDTH-1:0] vx_snp_req_tag;
wire vx_snp_req_ready;

reg vx_snp_rsp_valid;
`DEBUG_BEGIN
reg [`VX_SNP_TAG_WIDTH-1:0] vx_snp_rsp_tag;
`DEBUG_END
reg vx_snp_rsp_ready;

wire        vx_csr_io_req_valid;
wire [`VX_CSR_ID_WIDTH-1:0] vx_csr_io_req_coreid;
wire [11:0] vx_csr_io_req_addr;
wire        vx_csr_io_req_rw;
wire [31:0] vx_csr_io_req_data;
wire        vx_csr_io_req_ready;

wire        vx_csr_io_rsp_valid;
wire [31:0] vx_csr_io_rsp_data;
wire        vx_csr_io_rsp_ready;

reg vx_reset;
wire vx_busy;

// AVS Queues /////////////////////////////////////////////////////////////////

wire avs_rtq_push;
wire avs_rtq_pop;
`DEBUG_BEGIN
wire avs_rtq_empty;
wire avs_rtq_full;
`DEBUG_BEGIN

wire avs_rdq_push;
wire avs_rdq_pop;
t_local_mem_data avs_rdq_dout;
wire avs_rdq_empty;
`DEBUG_BEGIN
wire avs_rdq_full;
`DEBUG_END

// CMD variables //////////////////////////////////////////////////////////////

t_ccip_clAddr              cmd_io_addr;
reg [DRAM_ADDR_WIDTH-1:0] cmd_mem_addr;
reg [DRAM_ADDR_WIDTH-1:0] cmd_data_size;

`ifdef SCOPE
wire [63:0]               cmd_scope_rdata;
wire [63:0]               cmd_scope_wdata;  
wire                      cmd_scope_read;
wire                      cmd_scope_write;
`endif

reg [`VX_CSR_ID_WIDTH-1:0] cmd_csr_core;
reg [11:0]               cmd_csr_addr;
reg [31:0]               cmd_csr_rdata;  
reg [31:0]               cmd_csr_wdata;

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

wire [2:0] cmd_type = (cp2af_sRxPort.c0.mmioWrValid && (MMIO_CMD_TYPE == mmio_hdr.address)) ? 3'(cp2af_sRxPort.c0.data) : 3'h0;

`ifdef SCOPE
reg scope_start;
`endif

// disable assertions until reset
`ifndef VERILATOR
initial begin
  $assertoff;  
end
`endif

always @(posedge clk) begin
  if (reset) begin
  `ifndef VERILATOR
    $asserton; // enable assertions
  `endif

    mmio_tx.hdr         <= 0;
    mmio_tx.data        <= 0;
    mmio_tx.mmioRdValid <= 0;
    cmd_io_addr         <= 0;
    cmd_mem_addr        <= 0;
    cmd_data_size       <= 0;
  `ifdef SCOPE
    scope_start         <= 0;
  `endif
  end
  else begin
    mmio_tx.mmioRdValid <= 0;
    // serve MMIO write request
    if (cp2af_sRxPort.c0.mmioWrValid)
    begin
    `ifdef SCOPE
      scope_start <= 1;
    `endif
      case (mmio_hdr.address)
        MMIO_IO_ADDR: begin                     
          cmd_io_addr <= t_ccip_clAddr'(cp2af_sRxPort.c0.data);          
        `ifdef DBG_PRINT_OPAE 
          $display("%t: MMIO_IO_ADDR: addr=%0h, data=0x%0h", $time, mmio_hdr.address, t_ccip_clAddr'(cp2af_sRxPort.c0.data));
        `endif
        end
        MMIO_MEM_ADDR: begin          
          cmd_mem_addr <= t_local_mem_addr'(cp2af_sRxPort.c0.data);                  
        `ifdef DBG_PRINT_OPAE
          $display("%t: MMIO_MEM_ADDR: addr=%0h, data=0x%0h", $time, mmio_hdr.address, t_local_mem_addr'(cp2af_sRxPort.c0.data));
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
      mmio_tx.hdr.tid <= mmio_hdr.tid; // copy TID
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
      mmio_tx.mmioRdValid <= 1; // post response
    end
  end
end

// COMMAND FSM ////////////////////////////////////////////////////////////////

wire cmd_read_done;
wire cmd_write_done;
wire cmd_clflush_done;
wire cmd_csr_done;
wire cmd_run_done;

always @(posedge clk) begin
  if (reset) begin
    state     <= STATE_IDLE;    
    vx_reset  <= 0;    
  end
  else begin
    
    vx_reset <= 0;

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
          CMD_CLFLUSH: begin
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE CFLUSH: addr=%0h size=%0d", $time, cmd_mem_addr, cmd_data_size);
          `endif
            state <= STATE_CLFLUSH;
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

      STATE_START: begin // vortex reset cycle
        state <= STATE_RUN;
      end

      STATE_RUN: begin
        if (cmd_run_done) begin
          state <= STATE_IDLE;
        `ifdef DBG_PRINT_OPAE
          $display("%t: STATE IDLE", $time);
        `endif
        end
      end

      STATE_CLFLUSH: begin
        if (cmd_clflush_done) begin
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

wire vortex_enabled;
wire cci_rdq_empty;
wire [CCI_RD_RQ_DATAW-1:0] cci_rdq_dout;

wire cci_dram_rd_req_fire;
wire cci_dram_wr_req_fire;
wire vx_dram_rd_req_fire;
`DEBUG_BEGIN
wire vx_dram_wr_req_fire;
`DEBUG_END
wire vx_dram_rd_rsp_fire;

t_local_mem_byte_mask vx_dram_req_byteen_;
reg [$clog2(AVS_RD_QUEUE_SIZE+1)-1:0] avs_pending_reads;
wire [$clog2(AVS_RD_QUEUE_SIZE+1)-1:0] avs_pending_reads_next;
wire [DRAM_LINE_LW-1:0] vx_dram_req_offset, vx_dram_rsp_offset;
reg [DRAM_ADDR_WIDTH-1:0] cci_dram_rd_req_addr, cci_dram_wr_req_addr;

wire cci_dram_rd_req_enable, cci_dram_wr_req_enable;
wire vx_dram_req_enable, vx_dram_rd_req_enable, vx_dram_wr_req_enable;

reg [DRAM_ADDR_WIDTH-1:0] cci_dram_rd_req_ctr, cci_dram_wr_req_ctr;

assign vortex_enabled = (STATE_RUN == state) || (STATE_CLFLUSH == state);

assign cci_dram_rd_req_enable = (state == STATE_READ) 
                             && (avs_pending_reads < AVS_RD_QUEUE_SIZE)
                             && (cci_dram_rd_req_ctr != 0);

assign cci_dram_wr_req_enable = (state == STATE_WRITE)
                             && !cci_rdq_empty 
                             && (cci_dram_wr_req_ctr < cmd_data_size);

assign vx_dram_req_enable    = vortex_enabled && (avs_pending_reads < AVS_RD_QUEUE_SIZE);
assign vx_dram_rd_req_enable = vx_dram_req_enable && vx_dram_req_valid && !vx_dram_req_rw;
assign vx_dram_wr_req_enable = vx_dram_req_enable && vx_dram_req_valid && vx_dram_req_rw;

assign cci_dram_rd_req_fire = cci_dram_rd_req_enable && !avs_waitrequest;
assign cci_dram_wr_req_fire = cci_dram_wr_req_enable && !avs_waitrequest;

assign vx_dram_rd_req_fire  = vx_dram_rd_req_enable && !avs_waitrequest;
assign vx_dram_wr_req_fire  = vx_dram_wr_req_enable && !avs_waitrequest;

assign vx_dram_rd_rsp_fire  = vx_dram_rsp_valid && vx_dram_rsp_ready;

assign avs_pending_reads_next = avs_pending_reads 
                              + $bits(avs_pending_reads)'(((cci_dram_rd_req_fire || vx_dram_rd_req_fire) && !avs_rdq_pop) ? 1 :
                                                          (~(cci_dram_rd_req_fire || vx_dram_rd_req_fire) && avs_rdq_pop) ? -1 : 0);

if (`VX_DRAM_LINE_WIDTH != DRAM_LINE_WIDTH) begin
  assign vx_dram_req_offset  = ((DRAM_LINE_LW)'(vx_dram_req_addr[(DRAM_LINE_LW-VX_DRAM_LINE_LW)-1:0])) << VX_DRAM_LINE_LW;    
  assign vx_dram_req_byteen_ = 64'(vx_dram_req_byteen) << (6'(vx_dram_req_addr[(DRAM_LINE_LW-VX_DRAM_LINE_LW)-1:0]) << (VX_DRAM_LINE_LW - 3));
end else begin
  assign vx_dram_req_offset  = 0;
  assign vx_dram_req_byteen_ = vx_dram_req_byteen;
end

always @(*) begin
  case (state)
    CMD_MEM_READ:  avs_address = cci_dram_rd_req_addr;
    CMD_MEM_WRITE: avs_address = cci_dram_wr_req_addr + (DRAM_ADDR_WIDTH'(CCI_RD_RQ_TAGW'(cci_rdq_dout)));
    default:       avs_address = vx_dram_req_addr[`VX_DRAM_ADDR_WIDTH-1:`VX_DRAM_ADDR_WIDTH-DRAM_ADDR_WIDTH];
  endcase

  case (state)
    CMD_MEM_READ:  avs_byteenable = 64'hffffffffffffffff;
    CMD_MEM_WRITE: avs_byteenable = 64'hffffffffffffffff;
    default:       avs_byteenable = vx_dram_req_byteen_;
  endcase

  case (state)
    CMD_MEM_WRITE: avs_writedata = cci_rdq_dout[CCI_RD_RQ_DATAW-1:CCI_RD_RQ_TAGW];
    default:       avs_writedata = DRAM_LINE_WIDTH'(vx_dram_req_data) << vx_dram_req_offset;
  endcase
end

assign avs_read  = cci_dram_rd_req_enable || vx_dram_rd_req_enable;
assign avs_write = cci_dram_wr_req_enable || vx_dram_wr_req_enable;

assign cmd_write_done = (cci_dram_wr_req_ctr >= cmd_data_size);

always @(posedge clk) begin
  if (reset) 
  begin    
    mem_bank_select      <= 0;
    avs_burstcount       <= 1;
    cci_dram_rd_req_addr <= 0;
    cci_dram_wr_req_addr <= 0;
    cci_dram_rd_req_ctr  <= 0;
    cci_dram_wr_req_ctr  <= 0;    
    avs_pending_reads    <= 0;
  end
  else begin
    
    if (state == STATE_IDLE) begin
      if (CMD_MEM_READ == cmd_type) begin
        cci_dram_rd_req_addr <= cmd_mem_addr;
        cci_dram_rd_req_ctr  <= cmd_data_size;
      end 
      else if (CMD_MEM_WRITE == cmd_type) begin
        cci_dram_wr_req_addr <= cmd_mem_addr;
        cci_dram_wr_req_ctr  <= 0;
      end
    end

    if (cci_dram_rd_req_fire) begin
      cci_dram_rd_req_addr <= cci_dram_rd_req_addr + DRAM_ADDR_WIDTH'(1);       
      cci_dram_rd_req_ctr  <= cci_dram_rd_req_ctr - DRAM_ADDR_WIDTH'(1);  
    `ifdef DBG_PRINT_OPAE
      $display("%t: AVS Rd Req: addr=%0h, rem=%0d, pending=%0d", $time, `DRAM_TO_BYTE_ADDR(avs_address), (cci_dram_rd_req_ctr - 1), avs_pending_reads_next);
    `endif
    end

    if (cci_dram_wr_req_fire) begin                
      cci_dram_wr_req_addr <= cci_dram_wr_req_addr + ((CCI_RD_RQ_TAGW'(cci_dram_wr_req_ctr) == CCI_RD_RQ_TAGW'(CCI_RD_WINDOW_SIZE-1)) ? DRAM_ADDR_WIDTH'(CCI_RD_WINDOW_SIZE)  : DRAM_ADDR_WIDTH'(0));
      cci_dram_wr_req_ctr  <= cci_dram_wr_req_ctr + DRAM_ADDR_WIDTH'(1);
    `ifdef DBG_PRINT_OPAE
      $display("%t: AVS Wr Req: addr=%0h, data=%0h, rem=%0d", $time, `DRAM_TO_BYTE_ADDR(avs_address), avs_writedata, (cci_dram_wr_req_ctr + 1));
    `endif
    end    

    `ifdef DBG_PRINT_OPAE
      if (vx_dram_rd_req_fire) begin
        $display("%t: AVS Rd Req: addr=%0h, byteen=%0h, tag=%0h, pending=%0d", $time, `DRAM_TO_BYTE_ADDR(avs_address), avs_byteenable, vx_dram_req_tag, avs_pending_reads_next);
      end 
      
      if (vx_dram_wr_req_fire) begin
        $display("%t: AVS Wr Req: addr=%0h, byteen=%0h, tag=%0h, data=%0h", $time, `DRAM_TO_BYTE_ADDR(avs_address), avs_byteenable, vx_dram_req_tag, avs_writedata);
      end   

      if (avs_readdatavalid) begin
        $display("%t: AVS Rd Rsp: data=%0h, pending=%0d", $time, avs_readdata, avs_pending_reads_next);
      end
    `endif

    avs_pending_reads <= avs_pending_reads_next;   
  end
end

// Vortex DRAM requests

assign vx_dram_req_ready = vx_dram_req_enable && !avs_waitrequest;

// Vortex DRAM fill response

assign vx_dram_rsp_valid = vortex_enabled && !avs_rdq_empty;
if (`VX_DRAM_LINE_WIDTH != DRAM_LINE_WIDTH) begin
  assign vx_dram_rsp_data = (`VX_DRAM_LINE_WIDTH)'(avs_rdq_dout >> vx_dram_rsp_offset);
end else begin
  assign vx_dram_rsp_data = avs_rdq_dout;    
end

// AVS address read request queue /////////////////////////////////////////////

assign avs_rtq_push = vx_dram_rd_req_fire;
assign avs_rtq_pop  = vx_dram_rd_rsp_fire;

VX_generic_queue #(
  .DATAW(`VX_DRAM_TAG_WIDTH + DRAM_LINE_LW),
  .SIZE(AVS_RD_QUEUE_SIZE)
) avs_rd_req_queue (
  .clk      (clk),
  .reset    (reset),
  .push     (avs_rtq_push),
  .data_in  ({vx_dram_req_tag, vx_dram_req_offset}),
  .pop      (avs_rtq_pop),
  .data_out ({vx_dram_rsp_tag, vx_dram_rsp_offset}),
  .empty    (avs_rtq_empty),
  .full     (avs_rtq_full),
  `UNUSED_PIN (size)
);

// AVS data read response queue ///////////////////////////////////////////////

wire cci_wr_req_fire;

assign avs_rdq_push = avs_readdatavalid;
assign avs_rdq_pop  = vx_dram_rd_rsp_fire || cci_wr_req_fire; 

VX_generic_queue #(
  .DATAW(DRAM_LINE_WIDTH),
  .SIZE(AVS_RD_QUEUE_SIZE)
) avs_rd_rsp_queue (
  .clk      (clk),
  .reset    (reset),
  .push     (avs_rdq_push),
  .data_in  (avs_readdata),
  .pop      (avs_rdq_pop),
  .data_out (avs_rdq_dout),
  .empty    (avs_rdq_empty),
  .full     (avs_rdq_full),
  `UNUSED_PIN (size)
);

// CCI-P Read Request ///////////////////////////////////////////////////////////

reg [$clog2(CCI_RD_QUEUE_SIZE+1)-1:0] cci_pending_reads;
wire [$clog2(CCI_RD_QUEUE_SIZE+1)-1:0] cci_pending_reads_next;
reg [DRAM_ADDR_WIDTH-1:0] cci_rd_req_ctr;
wire [DRAM_ADDR_WIDTH-1:0] cci_rd_req_ctr_next;
wire [CCI_RD_RQ_TAGW-1:0] cci_rd_req_tag, cci_rd_rsp_tag;
reg [CCI_RD_RQ_TAGW-1:0] cci_rd_rsp_ctr;
t_ccip_clAddr cci_rd_req_addr;

wire cci_rd_req_fire, cci_rd_rsp_fire;
reg cci_rd_req_enable, cci_rd_req_wait;

wire cci_rdq_push, cci_rdq_pop;
wire [CCI_RD_RQ_DATAW-1:0] cci_rdq_din;

always @(*) begin
  af2cp_sTxPort.c0.hdr         = t_ccip_c0_ReqMemHdr'(0);
  af2cp_sTxPort.c0.hdr.address = cci_rd_req_addr;  
  af2cp_sTxPort.c0.hdr.mdata   = t_ccip_mdata'(cci_rd_req_tag);
end

assign cci_rd_req_fire = af2cp_sTxPort.c0.valid && !cp2af_sRxPort.c0TxAlmFull;
assign cci_rd_rsp_fire = (STATE_WRITE == state) && cp2af_sRxPort.c0.rspValid;

assign cci_rd_req_tag = CCI_RD_RQ_TAGW'(cci_rd_req_ctr);
assign cci_rd_rsp_tag = CCI_RD_RQ_TAGW'(cp2af_sRxPort.c0.hdr.mdata);

assign cci_rd_req_ctr_next = cci_rd_req_ctr + DRAM_ADDR_WIDTH'(cci_rd_req_fire ? 1 : 0);

assign cci_rdq_pop  = cci_dram_wr_req_fire;
assign cci_rdq_push = cci_rd_rsp_fire;
assign cci_rdq_din  = {cp2af_sRxPort.c0.data, cci_rd_rsp_tag};  

assign cci_pending_reads_next = cci_pending_reads 
                              + $bits(cci_pending_reads)'((cci_rd_req_fire && !cci_rdq_pop) ? 1 : 
                                                          (!cci_rd_req_fire && cci_rdq_pop) ? -1 : 0);

assign af2cp_sTxPort.c0.valid = cci_rd_req_enable && !cci_rd_req_wait;

// Send read requests to CCI
always @(posedge clk) begin
  if (reset) begin
    cci_rd_req_addr   <= 0;
    cci_rd_req_ctr    <= 0;
    cci_rd_rsp_ctr    <= 0;
    cci_pending_reads <= 0;
    cci_rd_req_enable <= 0;
    cci_rd_req_wait   <= 0;
  end 
  else begin      
    
    if ((STATE_IDLE == state) 
    &&  (CMD_MEM_WRITE == cmd_type)) begin
      cci_rd_req_addr   <= cmd_io_addr;
      cci_rd_req_ctr    <= 0;
      cci_rd_rsp_ctr    <= 0;
      cci_pending_reads <= 0;
      cci_rd_req_enable <= (cmd_data_size != 0);
      cci_rd_req_wait   <= 0;
    end

    cci_rd_req_enable <= (STATE_WRITE == state)                       
                      && (cci_rd_req_ctr_next < cmd_data_size)
                      && (cci_pending_reads_next < CCI_RD_QUEUE_SIZE);    

    if (cci_rd_req_fire) begin  
      cci_rd_req_addr <= cci_rd_req_addr + 1;
      cci_rd_req_ctr  <= cci_rd_req_ctr_next;
      if (cci_rd_req_tag == CCI_RD_RQ_TAGW'(CCI_RD_WINDOW_SIZE-1)) begin
        cci_rd_req_wait <= 1; // end current request batch
        $display("*** %t: CCI Rd Rsp: STOP", $time);
      end 
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Req: addr=%0h, tag=%0h, rem=%0d, pending=%0d", $time, cci_rd_req_addr, cci_rd_req_tag, (cmd_data_size - cci_rd_req_ctr_next), cci_pending_reads_next);
    `endif
    end

    if (cci_rd_rsp_fire) begin
      cci_rd_rsp_ctr <= cci_rd_rsp_ctr + CCI_RD_RQ_TAGW'(1);
      if (cci_rd_rsp_ctr == CCI_RD_RQ_TAGW'(CCI_RD_WINDOW_SIZE-1)) begin
        cci_rd_req_wait <= 0; // restart new request batch
        $display("*** %t: CCI Rd Rsp: START", $time);
      end 
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Rsp: idx=%0d, ctr=%0d", $time, cci_rd_rsp_tag, cci_rd_rsp_ctr);
    `endif
    end

    if (cci_rdq_pop) begin
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Queue Pop: pending=%0d", $time, cci_pending_reads_next);
    `endif
    end

    cci_pending_reads <= cci_pending_reads_next;
  end
end

VX_generic_queue #(
  .DATAW(CCI_RD_RQ_DATAW),
  .SIZE(CCI_RD_QUEUE_SIZE)
) cci_rd_req_queue (
  .clk      (clk),
  .reset    (reset),
  .push     (cci_rdq_push),
  .data_in  (cci_rdq_din),
  .pop      (cci_rdq_pop),
  .data_out (cci_rdq_dout),
  .empty    (cci_rdq_empty),
  `UNUSED_PIN (full),
  `UNUSED_PIN (size)
);

`DEBUG_BEGIN
reg [CCI_RD_WINDOW_SIZE-1:0] dbg_cci_rd_rsp_mask;
always @(posedge clk) begin
  if (reset) begin
    dbg_cci_rd_rsp_mask <= 0;
  end else begin
    if (cci_rd_rsp_fire) begin      
      if (cci_rd_rsp_ctr == 0) begin
        dbg_cci_rd_rsp_mask <= (CCI_RD_WINDOW_SIZE'(1) << cci_rd_rsp_tag);               
      end else begin                
        if (dbg_cci_rd_rsp_mask[cci_rd_rsp_tag] != 0) begin
          $display("*** %t: Assert: CCI Rd Rsp: idx=%0d, ctr=%0d, mask=%0h, meta=%0h, data=%0h", $time, cci_rd_rsp_tag, cci_rd_rsp_ctr, dbg_cci_rd_rsp_mask, cp2af_sRxPort.c0.hdr.mdata, cp2af_sRxPort.c0.data);
          assert(0);
        end
        dbg_cci_rd_rsp_mask[cci_rd_rsp_tag] <= 1;
      end 
    end
  end
end
`DEBUG_END

// CCI-P Write Request //////////////////////////////////////////////////////////

reg [$clog2(CCI_RW_QUEUE_SIZE+1)-1:0] cci_pending_writes;
wire [$clog2(CCI_RW_QUEUE_SIZE+1)-1:0] cci_pending_writes_next;
reg [DRAM_ADDR_WIDTH-1:0] cci_wr_req_ctr;
t_ccip_clAddr cci_wr_req_addr;
reg cci_wr_req_enable;
wire cci_wr_rsp_fire;

always @(*) begin
  af2cp_sTxPort.c1.hdr         = t_ccip_c1_ReqMemHdr'(0);
  af2cp_sTxPort.c1.hdr.address = cci_wr_req_addr;
  af2cp_sTxPort.c1.hdr.sop     = 1; // single line write mode
  af2cp_sTxPort.c1.data        = t_ccip_clData'(avs_rdq_dout);  
end 

assign cci_wr_req_fire = af2cp_sTxPort.c1.valid && !cp2af_sRxPort.c1TxAlmFull;
assign cci_wr_rsp_fire = (STATE_READ == state) && cp2af_sRxPort.c1.rspValid;

assign cci_pending_writes_next = cci_pending_writes 
                               + $bits(cci_pending_writes)'((cci_wr_req_fire && !cci_wr_rsp_fire) ? 1 :
                                                            (!cci_wr_req_fire && cci_wr_rsp_fire) ? -1 : 0);

assign cmd_read_done = (0 == cci_wr_req_ctr) && (0 == cci_pending_writes);

assign af2cp_sTxPort.c1.valid = cci_wr_req_enable && !avs_rdq_empty;

// Send write requests to CCI
always @(posedge clk) 
begin
  if (reset) begin
    cci_wr_req_addr    <= 0;
    cci_wr_req_ctr     <= 0;
    cci_wr_req_enable  <= 0;
    cci_pending_writes <= 0;
  end
  else begin
    
    if ((STATE_IDLE == state) 
    &&  (CMD_MEM_READ == cmd_type)) begin
      cci_wr_req_addr    <= cmd_io_addr;
      cci_wr_req_ctr     <= cmd_data_size;
      cci_pending_writes <= 0;
    end    

    cci_wr_req_enable <= (STATE_READ == state) 
                      && (cci_pending_writes_next < CCI_RW_QUEUE_SIZE);    

    if (cci_wr_req_fire) begin
      assert(cci_wr_req_ctr != 0);  
      cci_wr_req_addr <= cci_wr_req_addr + t_ccip_clAddr'(1);        
      cci_wr_req_ctr  <= cci_wr_req_ctr - DRAM_ADDR_WIDTH'(1);
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Wr Req: addr=%0h, rem=%0d, pending=%0d, data=%0h", $time, cci_wr_req_addr, (cci_wr_req_ctr - 1), cci_pending_writes_next, avs_rdq_dout);
    `endif
    end

  `ifdef DBG_PRINT_OPAE
    if (cci_wr_rsp_fire) begin      
      $display("%t: CCI Wr Rsp: pending=%0d", $time, cci_pending_writes_next);      
    end
  `endif

    cci_pending_writes <= cci_pending_writes_next;
  end
end

// Vortex cache snooping //////////////////////////////////////////////////////

wire [`VX_DRAM_ADDR_WIDTH-1:0] snp_req_size;
wire [`VX_DRAM_ADDR_WIDTH-1:0] snp_req_baseaddr;
reg [`VX_DRAM_ADDR_WIDTH-1:0] snp_req_ctr, snp_rsp_ctr;
wire [`VX_DRAM_ADDR_WIDTH-1:0] snp_req_ctr_next, snp_rsp_ctr_next;

wire vx_snp_req_fire, vx_snp_rsp_fire;

if (`VX_DRAM_LINE_WIDTH != DRAM_LINE_WIDTH) begin
  assign snp_req_baseaddr = {cmd_mem_addr, (`VX_DRAM_ADDR_WIDTH - DRAM_ADDR_WIDTH)'(0)};
  assign snp_req_size     = {cmd_data_size, (`VX_DRAM_ADDR_WIDTH - DRAM_ADDR_WIDTH)'(0)};
end else begin
  assign snp_req_baseaddr = cmd_mem_addr;
  assign snp_req_size     = cmd_data_size;
end

assign vx_snp_req_fire  = vx_snp_req_valid && vx_snp_req_ready;
assign vx_snp_rsp_fire  = vx_snp_rsp_valid && vx_snp_rsp_ready;

assign snp_req_ctr_next = vx_snp_req_fire ? (snp_req_ctr + `VX_DRAM_ADDR_WIDTH'(1)) : snp_req_ctr;
assign snp_rsp_ctr_next = vx_snp_rsp_fire ? (snp_rsp_ctr - `VX_DRAM_ADDR_WIDTH'(1)) : snp_rsp_ctr;

assign cmd_clflush_done = (0 == snp_rsp_ctr);  

always @(posedge clk) begin
  if (reset) begin
    vx_snp_req_valid <= 0;
    vx_snp_req_addr  <= 0;
    vx_snp_req_tag   <= 0;
    vx_snp_rsp_ready <= 0;
    snp_req_ctr      <= 0;
    snp_rsp_ctr      <= 0;
  end
  else begin

    if ((STATE_IDLE == state) 
    &&  (CMD_CLFLUSH == cmd_type)) begin
      vx_snp_req_addr  <= snp_req_baseaddr;
      vx_snp_req_tag   <= 0;
      snp_req_ctr      <= 0;
      snp_rsp_ctr      <= snp_req_size;
      vx_snp_req_valid <= (snp_req_size != 0);
      vx_snp_rsp_ready <= (snp_req_size != 0);
    end

    if ((STATE_CLFLUSH == state) 
     && (snp_req_ctr_next >= snp_req_size)) begin
       vx_snp_req_valid <= 0;
    end

    if ((STATE_CLFLUSH == state) 
     && (0 == snp_rsp_ctr_next)) begin
       vx_snp_rsp_ready <= 0;
    end

    if (vx_snp_req_fire)
    begin
      assert(snp_req_ctr < snp_req_size);
      vx_snp_req_addr <= vx_snp_req_addr + `VX_DRAM_ADDR_WIDTH'(1);
      vx_snp_req_tag  <= (`VX_SNP_TAG_WIDTH)'(snp_req_ctr_next);
      snp_req_ctr     <= snp_req_ctr_next;
    `ifdef DBG_PRINT_OPAE
      $display("%t: AFU Snp Req: addr=%0h, tag=%0d, rem=%0d", $time, `DRAM_TO_BYTE_ADDR(vx_snp_req_addr), (`VX_SNP_TAG_WIDTH)'(vx_snp_req_tag), (snp_req_size - snp_req_ctr_next));
    `endif
    end

    if ((STATE_CLFLUSH == state) 
     && vx_snp_rsp_fire) begin
       assert(snp_rsp_ctr != 0);
       snp_rsp_ctr <= snp_rsp_ctr_next;
    `ifdef DBG_PRINT_OPAE
      $display("%t: AFU Snp Rsp: tag=%0d, rem=%0d", $time, vx_snp_rsp_tag, snp_rsp_ctr_next);
    `endif
    end
  end
end

// CSRs///////////////////////////////////////////////////////////////////////

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
    cmd_csr_rdata   <= 0;
  end
  else begin
    if (vx_csr_io_req_valid && vx_csr_io_req_ready) begin
      csr_io_req_sent <= 1;
    end
    if (cmd_csr_done) begin
      csr_io_req_sent <= 0;
    end
    if ((STATE_CSR_READ == state) 
      && vx_csr_io_rsp_ready 
      && vx_csr_io_rsp_valid) begin
      cmd_csr_rdata <= vx_csr_io_rsp_data;
    end
  end
end

// Vortex /////////////////////////////////////////////////////////////////////

assign cmd_run_done = !vx_busy;

Vortex #() vortex (
  `SCOPE_BIND_top_vortex

  .clk              (clk),
  .reset            (reset | vx_reset),

  // DRAM request 
  .dram_req_valid   (vx_dram_req_valid),
  .dram_req_rw   	  (vx_dram_req_rw),
  .dram_req_byteen  (vx_dram_req_byteen),
  .dram_req_addr 		(vx_dram_req_addr),
  .dram_req_data		(vx_dram_req_data),
  .dram_req_tag     (vx_dram_req_tag),
  .dram_req_ready   (vx_dram_req_ready),

  // DRAM response  
  .dram_rsp_valid 	(vx_dram_rsp_valid),
  .dram_rsp_data	  (vx_dram_rsp_data),
  .dram_rsp_tag     (vx_dram_rsp_tag),
  .dram_rsp_ready   (vx_dram_rsp_ready),

  // Snoop request
  .snp_req_valid 	  (vx_snp_req_valid),
  .snp_req_addr     (vx_snp_req_addr),
  .snp_req_invalidate(vx_snp_req_invalidate),
  .snp_req_tag      (vx_snp_req_tag),
  .snp_req_ready    (vx_snp_req_ready),

  // Snoop response
  .snp_rsp_valid 	  (vx_snp_rsp_valid),
  .snp_rsp_tag      (vx_snp_rsp_tag),
  .snp_rsp_ready    (vx_snp_rsp_ready),

  // I/O request
  `UNUSED_PIN       (io_req_valid),
  `UNUSED_PIN       (io_req_rw),
  `UNUSED_PIN       (io_req_byteen),   
  `UNUSED_PIN       (io_req_addr),
  `UNUSED_PIN       (io_req_data),
  `UNUSED_PIN       (io_req_tag),    
  .io_req_ready     (1'b1),

  // I/O response
  .io_rsp_valid     (1'b0),
  .io_rsp_data      (0),
  .io_rsp_tag       (0),
  `UNUSED_PIN       (io_rsp_ready),

  // CSR I/O Request
  .csr_io_req_valid (vx_csr_io_req_valid),
  .csr_io_req_coreid(vx_csr_io_req_coreid),
  .csr_io_req_addr  (vx_csr_io_req_addr),
  .csr_io_req_rw    (vx_csr_io_req_rw),
  .csr_io_req_data  (vx_csr_io_req_data),
  .csr_io_req_ready (vx_csr_io_req_ready),

  // CSR I/O Response
  .csr_io_rsp_valid (vx_csr_io_rsp_valid),
  .csr_io_rsp_data  (vx_csr_io_rsp_data),
  .csr_io_rsp_ready (vx_csr_io_rsp_ready),
 
  // status
  .busy 				    (vx_busy),
  `UNUSED_PIN       (ebreak)
);

always @(posedge clk) begin
  if (!reset) begin
    // DRAM reads should only happen during vortex execution    
    assert(vx_busy || !vx_dram_rd_req_enable);
  end
end

// SCOPE //////////////////////////////////////////////////////////////////////

`ifdef SCOPE

`SCOPE_ASSIGN (scope_reset, vx_reset);

`SCOPE_ASSIGN (scope_dram_req_valid, vx_dram_req_valid);
`SCOPE_ASSIGN (scope_dram_req_addr,  {vx_dram_req_addr, 4'b0});
`SCOPE_ASSIGN (scope_dram_req_rw,    vx_dram_req_rw);
`SCOPE_ASSIGN (scope_dram_req_byteen,vx_dram_req_byteen);
`SCOPE_ASSIGN (scope_dram_req_data,  vx_dram_req_data);
`SCOPE_ASSIGN (scope_dram_req_tag,   vx_dram_req_tag);
`SCOPE_ASSIGN (scope_dram_req_ready, vx_dram_req_ready);

`SCOPE_ASSIGN (scope_dram_rsp_valid, vx_dram_rsp_valid);
`SCOPE_ASSIGN (scope_dram_rsp_data,  vx_dram_rsp_data);
`SCOPE_ASSIGN (scope_dram_rsp_tag,   vx_dram_rsp_tag);
`SCOPE_ASSIGN (scope_dram_rsp_ready, vx_dram_rsp_ready);

`SCOPE_ASSIGN (scope_snp_req_valid, vx_snp_req_valid);
`SCOPE_ASSIGN (scope_snp_req_addr,  {vx_snp_req_addr, 4'b0});
`SCOPE_ASSIGN (scope_snp_req_invalidate, vx_snp_req_invalidate);
`SCOPE_ASSIGN (scope_snp_req_tag,   vx_snp_req_tag);
`SCOPE_ASSIGN (scope_snp_req_ready, vx_snp_req_ready);

`SCOPE_ASSIGN (scope_snp_rsp_valid, vx_snp_rsp_valid);
`SCOPE_ASSIGN (scope_snp_rsp_tag,   vx_snp_rsp_tag);
`SCOPE_ASSIGN (scope_snp_rsp_ready, vx_snp_rsp_ready);

`SCOPE_ASSIGN (scope_snp_rsp_valid, vx_snp_rsp_valid);
`SCOPE_ASSIGN (scope_snp_rsp_tag,   vx_snp_rsp_tag);
`SCOPE_ASSIGN (scope_snp_rsp_ready, vx_snp_rsp_ready);

`SCOPE_ASSIGN (scope_busy, vx_busy);

wire scope_changed = `SCOPE_TRIGGER;

VX_scope #(
  .DATAW    ($bits({`SCOPE_DATA_LIST,`SCOPE_UPDATE_LIST})),
  .BUSW     (64),
  .SIZE     (4096),
  .UPDW     ($bits({`SCOPE_UPDATE_LIST}))
) scope (
  .clk      (clk),
  .reset    (reset),
  .start    (scope_start),
  .stop     (1'b0),
  .changed  (scope_changed),
  .data_in  ({`SCOPE_DATA_LIST,`SCOPE_UPDATE_LIST}),
  .bus_in   (cmd_scope_wdata),
  .bus_out  (cmd_scope_rdata),
  .bus_read (cmd_scope_read),
  .bus_write(cmd_scope_write)
);

`endif

endmodule