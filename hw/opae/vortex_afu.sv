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

`define VX_TO_DRAM_ADDR(x)    x[`VX_DRAM_ADDR_WIDTH-1:(`VX_DRAM_ADDR_WIDTH-DRAM_ADDR_WIDTH)] 

module vortex_afu #(
  parameter NUM_LOCAL_MEM_BANKS = 2
) (
  // global signals
  input clk,
  input SoftReset,

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

localparam CMD_TYPE_READ      = `AFU_IMAGE_CMD_TYPE_READ;
localparam CMD_TYPE_WRITE     = `AFU_IMAGE_CMD_TYPE_WRITE;
localparam CMD_TYPE_RUN       = `AFU_IMAGE_CMD_TYPE_RUN;
localparam CMD_TYPE_CLFLUSH   = `AFU_IMAGE_CMD_TYPE_CLFLUSH;

localparam MMIO_CSR_CMD       = `AFU_IMAGE_MMIO_CSR_CMD; 
localparam MMIO_CSR_IO_ADDR   = `AFU_IMAGE_MMIO_CSR_IO_ADDR;
localparam MMIO_CSR_MEM_ADDR  = `AFU_IMAGE_MMIO_CSR_MEM_ADDR;
localparam MMIO_CSR_DATA_SIZE = `AFU_IMAGE_MMIO_CSR_DATA_SIZE;
localparam MMIO_CSR_STATUS    = `AFU_IMAGE_MMIO_CSR_STATUS;

localparam MMIO_CSR_SCOPE_CMD = `AFU_IMAGE_MMIO_CSR_SCOPE_CMD;
localparam MMIO_CSR_SCOPE_DATA= `AFU_IMAGE_MMIO_CSR_SCOPE_DATA;

logic [127:0] afu_id = `AFU_ACCEL_UUID;

typedef enum logic[3:0] { 
  STATE_IDLE,
  STATE_READ,
  STATE_WRITE,
  STATE_START,
  STATE_RUN, 
  STATE_CLFLUSH
} state_t;

typedef logic [$clog2(CCI_RD_WINDOW_SIZE)-1:0] t_cci_rdq_tag;
typedef logic [$bits(t_ccip_clData) + $bits(t_cci_rdq_tag)-1:0] t_cci_rdq_data;

state_t state;

// Vortex ports ///////////////////////////////////////////////////////////////

logic vx_dram_req_valid;
logic vx_dram_req_rw;
logic [`VX_DRAM_BYTEEN_WIDTH-1:0] vx_dram_req_byteen;
logic [`VX_DRAM_ADDR_WIDTH-1:0] vx_dram_req_addr;
logic [`VX_DRAM_LINE_WIDTH-1:0] vx_dram_req_data;
logic [`VX_DRAM_TAG_WIDTH-1:0]  vx_dram_req_tag;
logic vx_dram_req_ready;

logic vx_dram_rsp_valid;
logic [`VX_DRAM_LINE_WIDTH-1:0] vx_dram_rsp_data;
logic [`VX_DRAM_TAG_WIDTH-1:0]  vx_dram_rsp_tag;
logic vx_dram_rsp_ready;

logic vx_snp_req_valid;
logic [`VX_DRAM_ADDR_WIDTH-1:0] vx_snp_req_addr;
logic vx_snp_req_invalidate = 0;
logic [`VX_SNP_TAG_WIDTH-1:0] vx_snp_req_tag;
logic vx_snp_req_ready;

logic vx_snp_rsp_valid;
`DEBUG_BEGIN
logic [`VX_SNP_TAG_WIDTH-1:0] vx_snp_rsp_tag;
`DEBUG_END
logic vx_snp_rsp_ready;

logic vx_reset;
logic vx_busy;

// AVS Queues /////////////////////////////////////////////////////////////////

logic avs_rtq_push;
logic avs_rtq_pop;
`DEBUG_BEGIN
logic avs_rtq_empty;
logic avs_rtq_full;
`DEBUG_BEGIN

logic avs_rdq_push;
logic avs_rdq_pop;
t_local_mem_data avs_rdq_dout;
logic avs_rdq_empty;
`DEBUG_BEGIN
logic avs_rdq_full;
`DEBUG_END

// CSR variables //////////////////////////////////////////////////////////////

logic [2:0]                csr_cmd;
t_ccip_clAddr              csr_io_addr;
logic[DRAM_ADDR_WIDTH-1:0] csr_mem_addr;
logic[DRAM_ADDR_WIDTH-1:0] csr_data_size;

logic [63:0]               csr_scope_cmd;  
logic [63:0]               csr_scope_data;
logic                      csr_scope_read;
logic                      csr_scope_write;

// MMIO controller ////////////////////////////////////////////////////////////

`IGNORE_WARNINGS_BEGIN
t_ccip_c0_ReqMmioHdr mmio_hdr; 
`IGNORE_WARNINGS_END
assign mmio_hdr = t_ccip_c0_ReqMmioHdr'(cp2af_sRxPort.c0.hdr);

t_if_ccip_c2_Tx mmio_tx;
assign af2cp_sTxPort.c2 = mmio_tx;

assign csr_scope_cmd   = 64'(cp2af_sRxPort.c0.data);
assign csr_scope_write = cp2af_sRxPort.c0.mmioWrValid && (MMIO_CSR_SCOPE_CMD == mmio_hdr.address);
assign csr_scope_read  = cp2af_sRxPort.c0.mmioRdValid && (MMIO_CSR_SCOPE_DATA == mmio_hdr.address);

always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    mmio_tx.hdr         <= 0;
    mmio_tx.data        <= 0;
    mmio_tx.mmioRdValid <= 0;
    csr_cmd             <= 0;
    csr_io_addr         <= 0;
    csr_mem_addr        <= 0;
    csr_data_size       <= 0;
  end
  else begin

    csr_cmd             <= 0;
    mmio_tx.mmioRdValid <= 0;

    // serve MMIO write request
    if (cp2af_sRxPort.c0.mmioWrValid)
    begin
      case (mmio_hdr.address)
        MMIO_CSR_IO_ADDR: begin                     
          csr_io_addr <= t_ccip_clAddr'(cp2af_sRxPort.c0.data);          
        `ifdef DBG_PRINT_OPAE 
          $display("%t: CSR_IO_ADDR: 0x%0h", $time, t_ccip_clAddr'(cp2af_sRxPort.c0.data));
        `endif
        end
        MMIO_CSR_MEM_ADDR: begin          
          csr_mem_addr <= t_local_mem_addr'(cp2af_sRxPort.c0.data);                  
        `ifdef DBG_PRINT_OPAE
          $display("%t: CSR_MEM_ADDR: 0x%0h", $time, t_local_mem_addr'(cp2af_sRxPort.c0.data));
        `endif
        end
        MMIO_CSR_DATA_SIZE: begin          
          csr_data_size <= $bits(csr_data_size)'(cp2af_sRxPort.c0.data);          
        `ifdef DBG_PRINT_OPAE
          $display("%t: CSR_DATA_SIZE: %0d", $time, $bits(csr_data_size)'(cp2af_sRxPort.c0.data));
        `endif
        end
        MMIO_CSR_CMD: begin          
          csr_cmd <= $bits(csr_cmd)'(cp2af_sRxPort.c0.data);
        `ifdef DBG_PRINT_OPAE
          $display("%t: CSR_CMD: %0d", $time, $bits(csr_cmd)'(cp2af_sRxPort.c0.data));
        `endif
        end
        MMIO_CSR_SCOPE_CMD: begin          
        `ifdef DBG_PRINT_OPAE
          $display("%t: CSR_SCOPE_CMD: %0d", $time, 64'(cp2af_sRxPort.c0.data));
        `endif
        end
        default: begin
           // user-defined CSRs
           //if (mmio_hdr.addres >= MMIO_CSR_USER) begin
             // write Vortex CRS
           //end
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
        MMIO_CSR_STATUS: begin
        `ifdef DBG_PRINT_OPAE
          if (state != mmio_tx.data) begin
            $display("%t: STATUS: state=%0d", $time, state);
          end
        `endif
          mmio_tx.data <= 64'(state);
        end  
        MMIO_CSR_SCOPE_DATA: begin          
          mmio_tx.data <= csr_scope_data;
        `ifdef DBG_PRINT_OPAE
          $display("%t: SCOPE: data=%0d", $time, csr_scope_data);
        `endif
        end
        default: mmio_tx.data <= 64'h0;
      endcase
      mmio_tx.mmioRdValid <= 1; // post response
    end
  end
end

// COMMAND FSM ////////////////////////////////////////////////////////////////

logic cmd_read_done;
logic cmd_write_done;
logic cmd_clflush_done;
logic cmd_run_done;

always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    state     <= STATE_IDLE;    
    vx_reset  <= 0;    
  end
  else begin
    
    vx_reset <= 0;

    case (state)
      STATE_IDLE: begin             
        case (csr_cmd)
          CMD_TYPE_READ: begin     
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE READ: ia=%0h da=%0h sz=%0d", $time, csr_io_addr, csr_mem_addr, csr_data_size);
          `endif
            state <= STATE_READ;   
          end 
          CMD_TYPE_WRITE: begin      
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE WRITE: ia=%0h da=%0h sz=%0d", $time, csr_io_addr, csr_mem_addr, csr_data_size);
          `endif
            state <= STATE_WRITE;
          end
          CMD_TYPE_RUN: begin        
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE START", $time);
          `endif
            vx_reset <= 1;
            state <= STATE_START;                    
          end
          CMD_TYPE_CLFLUSH: begin
          `ifdef DBG_PRINT_OPAE
            $display("%t: STATE CFLUSH: da=%0h sz=%0d", $time, csr_mem_addr, csr_data_size);
          `endif
            state <= STATE_CLFLUSH;
          end
          default: begin
            state <= state;
          end
        endcase
      end      

      STATE_READ: begin
        if (cmd_read_done) begin
          state <= STATE_IDLE;
        end
      end

      STATE_WRITE: begin
        if (cmd_write_done) begin
          state <= STATE_IDLE;
        end
      end

      STATE_START: begin // vortex reset cycle
        state <= STATE_RUN; 
      end

      STATE_RUN: begin
        if (cmd_run_done) begin
          state <= STATE_IDLE;
        end
      end

      STATE_CLFLUSH: begin
        if (cmd_clflush_done) begin
          state <= STATE_IDLE;
        end
      end

      default: begin
        state <= state;
      end

    endcase
  end
end

// AVS Controller /////////////////////////////////////////////////////////////

logic vortex_enabled;
logic cci_rdq_empty;
t_cci_rdq_data cci_rdq_dout;

logic cci_dram_rd_req_fire;
logic cci_dram_wr_req_fire;
logic vx_dram_rd_req_fire;
`DEBUG_BEGIN
logic vx_dram_wr_req_fire;
`DEBUG_END
logic vx_dram_rd_rsp_fire;

t_local_mem_byte_mask vx_dram_req_byteen_;
logic [$clog2(AVS_RD_QUEUE_SIZE+1)-1:0] avs_pending_reads, avs_pending_reads_next;
logic [DRAM_LINE_LW-1:0] vx_dram_req_offset, vx_dram_rsp_offset;
logic [DRAM_ADDR_WIDTH-1:0] cci_dram_rd_req_addr, cci_dram_wr_req_addr;

logic cci_dram_rd_req_enable, cci_dram_wr_req_enable;
logic vx_dram_req_enable, vx_dram_rd_req_enable, vx_dram_wr_req_enable;

logic [DRAM_ADDR_WIDTH-1:0] cci_dram_rd_req_ctr, cci_dram_wr_req_ctr;

assign vortex_enabled = (STATE_RUN == state) || (STATE_CLFLUSH == state);

assign cci_dram_rd_req_enable = (state == STATE_READ) 
                             && (avs_pending_reads < AVS_RD_QUEUE_SIZE)
                             && (cci_dram_rd_req_ctr != 0);

assign cci_dram_wr_req_enable = (state == STATE_WRITE)
                             && !cci_rdq_empty 
                             && (cci_dram_wr_req_ctr < csr_data_size);

assign vx_dram_req_enable    = vortex_enabled && (avs_pending_reads < AVS_RD_QUEUE_SIZE);
assign vx_dram_rd_req_enable = vx_dram_req_enable && vx_dram_req_valid && ~vx_dram_req_rw;
assign vx_dram_wr_req_enable = vx_dram_req_enable && vx_dram_req_valid && vx_dram_req_rw;

assign cci_dram_rd_req_fire = cci_dram_rd_req_enable && ~avs_waitrequest;
assign cci_dram_wr_req_fire = cci_dram_wr_req_enable && ~avs_waitrequest;

assign vx_dram_rd_req_fire  = vx_dram_rd_req_enable && ~avs_waitrequest;
assign vx_dram_wr_req_fire  = vx_dram_wr_req_enable && ~avs_waitrequest;

assign vx_dram_rd_rsp_fire  = vx_dram_rsp_valid && vx_dram_rsp_ready;

assign avs_pending_reads_next = avs_pending_reads 
                              + (((cci_dram_rd_req_fire || vx_dram_rd_req_fire) && ~avs_rdq_pop) ? 1 :
                                 (~(cci_dram_rd_req_fire || vx_dram_rd_req_fire) && avs_rdq_pop) ? -1 : 0);

if (`VX_DRAM_LINE_WIDTH != DRAM_LINE_WIDTH) begin
  assign vx_dram_req_offset  = ((DRAM_LINE_LW)'(vx_dram_req_addr[(DRAM_LINE_LW-VX_DRAM_LINE_LW)-1:0])) << VX_DRAM_LINE_LW;    
  assign vx_dram_req_byteen_ = 64'(vx_dram_req_byteen) << (6'(vx_dram_req_addr[(DRAM_LINE_LW-VX_DRAM_LINE_LW)-1:0]) << (VX_DRAM_LINE_LW - 3));
end else begin
  assign vx_dram_req_offset  = 0;
  assign vx_dram_req_byteen_ = vx_dram_req_byteen;
end

always_comb 
begin        
  case (state)
    CMD_TYPE_READ:  avs_address = cci_dram_rd_req_addr;
    CMD_TYPE_WRITE: avs_address = cci_dram_wr_req_addr + ((DRAM_ADDR_WIDTH)'(t_cci_rdq_tag'(cci_rdq_dout)));
    default:        avs_address = `VX_TO_DRAM_ADDR(vx_dram_req_addr);
  endcase

  case (state)
    CMD_TYPE_READ:  avs_byteenable = 64'hffffffffffffffff;
    CMD_TYPE_WRITE: avs_byteenable = 64'hffffffffffffffff;
    default:        avs_byteenable = vx_dram_req_byteen_;
  endcase

  case (state)
    CMD_TYPE_WRITE: avs_writedata = cci_rdq_dout[$bits(t_ccip_clData) + $bits(t_cci_rdq_tag)-1:$bits(t_cci_rdq_tag)];
    default:        avs_writedata = (DRAM_LINE_WIDTH)'(vx_dram_req_data) << vx_dram_req_offset;
  endcase
end

assign avs_read  = cci_dram_rd_req_enable || vx_dram_rd_req_enable;
assign avs_write = cci_dram_wr_req_enable || vx_dram_wr_req_enable;

assign cmd_write_done = (cci_dram_wr_req_ctr >= csr_data_size);

always_ff @(posedge clk) 
begin
  if (SoftReset) 
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
      if (CMD_TYPE_READ == csr_cmd) begin
        cci_dram_rd_req_addr <= csr_mem_addr;
        cci_dram_rd_req_ctr  <= csr_data_size;
      end 
      else if (CMD_TYPE_WRITE == csr_cmd) begin
        cci_dram_wr_req_addr <= csr_mem_addr;
        cci_dram_wr_req_ctr  <= 0;
      end
    end

    if (cci_dram_rd_req_fire) begin
      cci_dram_rd_req_addr <= cci_dram_rd_req_addr + 1;       
      cci_dram_rd_req_ctr  <= cci_dram_rd_req_ctr - 1;  
    `ifdef DBG_PRINT_OPAE
      $display("%t: AVS Rd Req: addr=%0h, rem=%0d, pending=%0d", $time, `DRAM_TO_BYTE_ADDR(avs_address), (cci_dram_rd_req_ctr - 1), avs_pending_reads_next);
    `endif
    end

    if (cci_dram_wr_req_fire) begin                
      cci_dram_wr_req_addr <= cci_dram_wr_req_addr + ((t_cci_rdq_tag'(cci_dram_wr_req_ctr) == t_cci_rdq_tag'(CCI_RD_WINDOW_SIZE-1)) ? (DRAM_ADDR_WIDTH)'(CCI_RD_WINDOW_SIZE)  : 0);
      cci_dram_wr_req_ctr  <= cci_dram_wr_req_ctr + 1;
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
  .reset    (SoftReset),
  .push     (avs_rtq_push),
  .data_in  ({vx_dram_req_tag, vx_dram_req_offset}),
  .pop      (avs_rtq_pop),
  .data_out ({vx_dram_rsp_tag, vx_dram_rsp_offset}),
  .empty    (avs_rtq_empty),
  .full     (avs_rtq_full),
  `UNUSED_PIN (size)
);

// AVS data read response queue ///////////////////////////////////////////////

logic cci_wr_req_fire;

assign avs_rdq_push = avs_readdatavalid;
assign avs_rdq_pop  = vx_dram_rd_rsp_fire || cci_wr_req_fire; 

VX_generic_queue #(
  .DATAW(DRAM_LINE_WIDTH),
  .SIZE(AVS_RD_QUEUE_SIZE)
) avs_rd_rsp_queue (
  .clk      (clk),
  .reset    (SoftReset),
  .push     (avs_rdq_push),
  .data_in  (avs_readdata),
  .pop      (avs_rdq_pop),
  .data_out (avs_rdq_dout),
  .empty    (avs_rdq_empty),
  .full     (avs_rdq_full),
  `UNUSED_PIN (size)
);

// CCI-P Read Request ///////////////////////////////////////////////////////////

logic [$clog2(CCI_RD_QUEUE_SIZE+1)-1:0] cci_pending_reads, cci_pending_reads_next;
logic [DRAM_ADDR_WIDTH-1:0] cci_rd_req_ctr, cci_rd_req_ctr_next;
t_ccip_clAddr cci_rd_req_addr;
t_cci_rdq_tag cci_rd_rsp_ctr;

logic cci_rd_req_fire, cci_rd_rsp_fire;
logic cci_rd_req_enable, cci_rd_req_wait;

logic cci_rdq_push, cci_rdq_pop;
t_cci_rdq_data cci_rdq_din;

always_comb begin
  af2cp_sTxPort.c0.hdr         = t_ccip_c0_ReqMemHdr'(0);
  af2cp_sTxPort.c0.hdr.address = cci_rd_req_addr;  
  af2cp_sTxPort.c0.hdr.mdata   = t_ccip_mdata'(t_cci_rdq_tag'(cci_rd_req_ctr));
end

assign cci_rd_req_fire = af2cp_sTxPort.c0.valid && !cp2af_sRxPort.c0TxAlmFull;
assign cci_rd_rsp_fire = (STATE_WRITE == state) && cp2af_sRxPort.c0.rspValid;

assign cci_rd_req_ctr_next = cci_rd_req_ctr + (cci_rd_req_fire ? 1 : 0);

assign cci_rdq_pop  = cci_dram_wr_req_fire;
assign cci_rdq_push = cci_rd_rsp_fire;
assign cci_rdq_din  = {cp2af_sRxPort.c0.data, t_cci_rdq_tag'(cp2af_sRxPort.c0.hdr.mdata)};  

assign cci_pending_reads_next = cci_pending_reads 
                              + ((cci_rd_req_fire && ~cci_rdq_pop) ? 1 : 
                                 (~cci_rd_req_fire && cci_rdq_pop) ? -1 : 0);

assign af2cp_sTxPort.c0.valid = cci_rd_req_enable && ~cci_rd_req_wait;

// Send read requests to CCI
always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    cci_rd_req_addr   <= 0;
    cci_rd_req_ctr    <= 0;
    cci_rd_rsp_ctr    <= 0;
    cci_pending_reads <= 0;
    cci_rd_req_enable <= 0;
    cci_rd_req_wait   <= 0;
  end 
  else begin      
    
    if ((STATE_IDLE == state) 
    &&  (CMD_TYPE_WRITE == csr_cmd)) begin
      cci_rd_req_addr   <= csr_io_addr;
      cci_rd_req_ctr    <= 0;
      cci_rd_rsp_ctr    <= 0;
      cci_pending_reads <= 0;
      cci_rd_req_enable <= (csr_data_size != 0);
      cci_rd_req_wait   <= 0;
    end

    cci_rd_req_enable <= (STATE_WRITE == state)                       
                      && (cci_rd_req_ctr_next < csr_data_size)
                      && (cci_pending_reads_next < CCI_RD_QUEUE_SIZE);    

    if (cci_rd_req_fire) begin  
      cci_rd_req_addr <= cci_rd_req_addr + 1;
      cci_rd_req_ctr  <= cci_rd_req_ctr_next;
      if (t_cci_rdq_tag'(cci_rd_req_ctr) == t_cci_rdq_tag'(CCI_RD_WINDOW_SIZE-1)) begin
        cci_rd_req_wait <= 1;   // end current request batch
      end 
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Req: addr=%0h, rem=%0d, pending=%0d", $time, cci_rd_req_addr, (csr_data_size - cci_rd_req_ctr_next), cci_pending_reads_next);
    `endif
    end

    if (cci_rd_rsp_fire) begin
      cci_rd_rsp_ctr <= cci_rd_rsp_ctr + 1;
      if (cci_rd_rsp_ctr == t_cci_rdq_tag'(CCI_RD_WINDOW_SIZE-1)) begin
        cci_rd_req_wait <= 0;   // restart new request batch
      end 
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Rd Rsp: idx=%0d, ctr=%0d", $time, t_cci_rdq_tag'(cp2af_sRxPort.c0.hdr.mdata), cci_rd_rsp_ctr);
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
  .DATAW($bits(t_ccip_clData) + $bits(t_cci_rdq_tag)),
  .SIZE(CCI_RD_QUEUE_SIZE)
) cci_rd_req_queue (
  .clk      (clk),
  .reset    (SoftReset),
  .push     (cci_rdq_push),
  .data_in  (cci_rdq_din),
  .pop      (cci_rdq_pop),
  .data_out (cci_rdq_dout),
  .empty    (cci_rdq_empty),
  `UNUSED_PIN (full),
  `UNUSED_PIN (size)
);

// CCI-P Write Request //////////////////////////////////////////////////////////

logic [$clog2(CCI_RW_QUEUE_SIZE+1)-1:0] cci_pending_writes, cci_pending_writes_next;
logic [DRAM_ADDR_WIDTH-1:0] cci_wr_req_ctr;
t_ccip_clAddr cci_wr_req_addr;
logic cci_wr_req_enable, cci_wr_rsp_fire;

always_comb begin
  af2cp_sTxPort.c1.hdr         = t_ccip_c1_ReqMemHdr'(0);
  af2cp_sTxPort.c1.hdr.address = cci_wr_req_addr;
  af2cp_sTxPort.c1.hdr.sop     = 1; // single line write mode
  af2cp_sTxPort.c1.data        = t_ccip_clData'(avs_rdq_dout);  
end 

assign cci_wr_req_fire = af2cp_sTxPort.c1.valid && !cp2af_sRxPort.c1TxAlmFull;
assign cci_wr_rsp_fire = (STATE_READ == state) && cp2af_sRxPort.c1.rspValid;

assign cci_pending_writes_next = cci_pending_writes 
                               + ((cci_wr_req_fire && ~cci_wr_rsp_fire) ? 1 :
                                  (~cci_wr_req_fire && cci_wr_rsp_fire) ? -1 : 0);

assign cmd_read_done = (0 == cci_wr_req_ctr) && (0 == cci_pending_writes);

assign af2cp_sTxPort.c1.valid = cci_wr_req_enable && ~avs_rdq_empty;

// Send write requests to CCI
always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    cci_wr_req_addr    <= 0;
    cci_wr_req_ctr     <= 0;
    cci_wr_req_enable  <= 0;
    cci_pending_writes <= 0;
  end
  else begin
    
    if ((STATE_IDLE == state) 
    &&  (CMD_TYPE_READ == csr_cmd)) begin
      cci_wr_req_addr    <= csr_io_addr;
      cci_wr_req_ctr     <= csr_data_size;
      cci_pending_writes <= 0;
    end    

    cci_wr_req_enable <= (STATE_READ == state) 
                      && (cci_pending_writes_next < CCI_RW_QUEUE_SIZE);    

    if (cci_wr_req_fire) begin
      assert(cci_wr_req_ctr != 0);  
      cci_wr_req_addr <= cci_wr_req_addr + 1;        
      cci_wr_req_ctr  <= cci_wr_req_ctr - 1;
    `ifdef DBG_PRINT_OPAE
      $display("%t: CCI Wr Req: addr=%0h, rem=%0d, pending=%0d", $time, cci_wr_req_addr, (cci_wr_req_ctr - 1), cci_pending_writes_next);
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

logic [`VX_DRAM_ADDR_WIDTH-1:0] snp_req_size;
logic [`VX_DRAM_ADDR_WIDTH-1:0] snp_req_baseaddr;
logic [`VX_DRAM_ADDR_WIDTH-1:0] snp_req_ctr, snp_req_ctr_next;
logic [`VX_DRAM_ADDR_WIDTH-1:0] snp_rsp_ctr, snp_rsp_ctr_next;

logic vx_snp_req_fire, vx_snp_rsp_fire;

if (`VX_DRAM_LINE_WIDTH != DRAM_LINE_WIDTH) begin
  assign snp_req_baseaddr = {csr_mem_addr, (`VX_DRAM_ADDR_WIDTH - DRAM_ADDR_WIDTH)'(0)};
  assign snp_req_size     = {csr_data_size, (`VX_DRAM_ADDR_WIDTH - DRAM_ADDR_WIDTH)'(0)};
end else begin
  assign snp_req_baseaddr = csr_mem_addr;
  assign snp_req_size     = csr_data_size;
end

assign vx_snp_req_fire  = vx_snp_req_valid && vx_snp_req_ready;
assign vx_snp_rsp_fire  = vx_snp_rsp_valid && vx_snp_rsp_ready;

assign snp_req_ctr_next = vx_snp_req_fire ? (snp_req_ctr + 1) : snp_req_ctr;
assign snp_rsp_ctr_next = vx_snp_rsp_fire ? (snp_rsp_ctr - 1) : snp_rsp_ctr;

assign cmd_clflush_done = (0 == snp_rsp_ctr);  

always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    vx_snp_req_valid <= 0;
    vx_snp_req_addr  <= 0;
    vx_snp_req_tag   <= 0;
    vx_snp_rsp_ready <= 0;
    snp_req_ctr      <= 0;
    snp_rsp_ctr      <= 0;
  end
  else begin

    if ((STATE_IDLE == state) 
    &&  (CMD_TYPE_CLFLUSH == csr_cmd)) begin
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
      vx_snp_req_addr <= vx_snp_req_addr + 1;
      vx_snp_req_tag  <= (`VX_SNP_TAG_WIDTH)'(snp_req_ctr_next);
      snp_req_ctr     <= snp_req_ctr_next;
    `ifdef DBG_PRINT_OPAE
      $display("%t: AFU Snp Req: addr=%0h, tag=%0d, rem=%0d", $time, `DRAM_TO_BYTE_ADDR(vx_snp_req_addr), (`VX_SNP_TAG_WIDTH)'(snp_req_ctr_next), (snp_req_size - snp_req_ctr_next));
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

// SCOPE //////////////////////////////////////////////////////////////////////

`ifdef SCOPE

`SCOPE_SIGNALS_DECL

`SCOPE_ASSIGN(scope_dram_req_valid, vx_dram_req_valid);
`SCOPE_ASSIGN(scope_dram_req_addr,  {vx_dram_req_addr, 4'b0});
`SCOPE_ASSIGN(scope_dram_req_tag,   vx_dram_req_tag);
`SCOPE_ASSIGN(scope_dram_req_ready, vx_dram_req_ready);
`SCOPE_ASSIGN(scope_dram_rsp_valid, vx_dram_rsp_valid);
`SCOPE_ASSIGN(scope_dram_rsp_tag,   vx_dram_rsp_tag);
`SCOPE_ASSIGN(scope_dram_rsp_ready, vx_dram_rsp_ready);

`SCOPE_ASSIGN(scope_snp_req_valid, vx_snp_req_valid);
`SCOPE_ASSIGN(scope_snp_req_addr,  {vx_snp_req_addr, 4'b0});
`SCOPE_ASSIGN(scope_snp_req_invalidate, vx_snp_req_invalidate);
`SCOPE_ASSIGN(scope_snp_req_tag,   vx_snp_req_tag);
`SCOPE_ASSIGN(scope_snp_req_ready, vx_snp_req_ready);
`SCOPE_ASSIGN(scope_snp_rsp_valid, vx_snp_rsp_valid);
`SCOPE_ASSIGN(scope_snp_rsp_tag,   vx_snp_rsp_tag);
`SCOPE_ASSIGN(scope_snp_rsp_ready, vx_snp_rsp_ready);

`STATIC_ASSERT($bits({`SCOPE_SIGNALS_DATA_LIST `SCOPE_SIGNALS_UPD_LIST}) == 490, "oops!")

wire force_changed = (scope_icache_req_valid && scope_icache_req_ready)
                  || (scope_icache_rsp_valid && scope_icache_rsp_ready)
                  || ((| scope_dcache_req_valid) && scope_dcache_req_ready)
                  || ((| scope_dcache_rsp_valid) && scope_dcache_rsp_ready)
                  || (scope_dram_req_valid && scope_dram_req_ready)
                  || (scope_dram_rsp_valid && scope_dram_rsp_ready)
                  || (scope_snp_req_valid && scope_snp_req_ready)
                  || (scope_snp_rsp_valid && scope_snp_rsp_ready);

VX_scope #(
  .DATAW    ($bits({`SCOPE_SIGNALS_DATA_LIST `SCOPE_SIGNALS_UPD_LIST})),
  .BUSW     (64),
  .SIZE     (4096),
  .UPDW     ($bits({`SCOPE_SIGNALS_UPD_LIST}))
) scope (
  .clk      (clk),
  .reset    (SoftReset),
  .start    (vx_reset),
  .stop     (cmd_run_done),
  .changed  (force_changed),
  .data_in  ({`SCOPE_SIGNALS_DATA_LIST `SCOPE_SIGNALS_UPD_LIST}),
  .bus_in   (csr_scope_cmd),
  .bus_out  (csr_scope_data),
  .bus_read (csr_scope_read),
  .bus_write(csr_scope_write)
);

`endif

// Vortex binding /////////////////////////////////////////////////////////////

assign cmd_run_done = !vx_busy;

Vortex_Socket #() vx_socket (
  `SCOPE_SIGNALS_ICACHE_ATTACH
  `SCOPE_SIGNALS_DCACHE_ATTACH
  `SCOPE_SIGNALS_CORE_ATTACH
  `SCOPE_SIGNALS_BE_ATTACH

  .clk              (clk),
  .reset            (vx_reset),

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
  .io_req_ready     (1),

  // I/O response
  .io_rsp_valid     (0),
  .io_rsp_data      (0),
  .io_rsp_tag       (0),
  `UNUSED_PIN       (io_rsp_ready),
 
  // status
  .busy 				    (vx_busy),
  `UNUSED_PIN       (ebreak)
);

endmodule