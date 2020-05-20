`include "platform_if.vh"
import local_mem_cfg_pkg::*;
`include "afu_json_info.vh"
`include "VX_define.vh"

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

localparam DRAM_ADDR_WIDTH = $bits(t_local_mem_addr);
localparam DRAM_LINE_WIDTH = $bits(t_local_mem_data);
localparam DRAM_TAG_WIDTH  = `L3DRAM_TAG_WIDTH;

`STATIC_ASSERT(DRAM_ADDR_WIDTH == `L3DRAM_ADDR_WIDTH, "invalid vortex dram bus!")
`STATIC_ASSERT(DRAM_LINE_WIDTH == `L3DRAM_LINE_WIDTH, "invalid vortex dram bus!")

localparam AVS_RD_QUEUE_SIZE  = 16;

localparam CCI_RD_WINDOW_SIZE = 8;
localparam CCI_RD_QUEUE_SIZE  = 2 * CCI_RD_WINDOW_SIZE;

localparam AFU_ID_L           = 16'h0002;      // AFU ID Lower
localparam AFU_ID_H           = 16'h0004;      // AFU ID Higher 

localparam CMD_TYPE_READ      = `AFU_IMAGE_CMD_TYPE_READ;
localparam CMD_TYPE_WRITE     = `AFU_IMAGE_CMD_TYPE_WRITE;
localparam CMD_TYPE_RUN       = `AFU_IMAGE_CMD_TYPE_RUN;
localparam CMD_TYPE_CLFLUSH   = `AFU_IMAGE_CMD_TYPE_CLFLUSH;

localparam MMIO_CSR_CMD       = `AFU_IMAGE_MMIO_CSR_CMD; 
localparam MMIO_CSR_STATUS    = `AFU_IMAGE_MMIO_CSR_STATUS;
localparam MMIO_CSR_IO_ADDR   = `AFU_IMAGE_MMIO_CSR_IO_ADDR;
localparam MMIO_CSR_MEM_ADDR  = `AFU_IMAGE_MMIO_CSR_MEM_ADDR;
localparam MMIO_CSR_DATA_SIZE = `AFU_IMAGE_MMIO_CSR_DATA_SIZE;

logic [127:0] afu_id = `AFU_ACCEL_UUID;

typedef enum logic[3:0] { 
  STATE_IDLE,
  STATE_READ,
  STATE_WRITE,
  STATE_START,
  STATE_RUN, 
  STATE_CLFLUSH
} state_t;

typedef logic [`LOG2UP(CCI_RD_WINDOW_SIZE)-1:0] t_cci_rdq_tag;
typedef logic [$bits(t_ccip_clData) + $bits(t_cci_rdq_tag)-1:0] t_cci_rdq_data;

state_t state;

// Vortex ports ///////////////////////////////////////////////////////////////

logic vx_dram_req_read;
logic vx_dram_req_write;
logic [DRAM_ADDR_WIDTH-1:0] vx_dram_req_addr;
logic [DRAM_LINE_WIDTH-1:0] vx_dram_req_data;
logic [DRAM_TAG_WIDTH-1:0]  vx_dram_req_tag;
logic vx_dram_req_ready;

logic vx_dram_rsp_valid;
logic [DRAM_LINE_WIDTH-1:0] vx_dram_rsp_data;
logic [DRAM_TAG_WIDTH-1:0]  vx_dram_rsp_tag;
logic vx_dram_rsp_ready;

logic vx_snp_req_valid;
logic [DRAM_ADDR_WIDTH-1:0] vx_snp_req_addr;
logic [0:0] vx_snp_req_tag;
logic vx_snp_req_ready;

logic vx_snp_rsp_valid;
logic [0:0] vx_snp_rsp_addr;
logic vx_snp_rsp_ready;

logic vx_busy;

// AVS Queues /////////////////////////////////////////////////////////////////

logic avs_rtq_push;
logic [DRAM_TAG_WIDTH-1:0] avs_rtq_din;
logic avs_rtq_pop;
logic [DRAM_TAG_WIDTH-1:0] avs_rtq_dout;
logic avs_rtq_empty;
logic avs_rtq_full;

logic avs_rdq_push;
t_local_mem_data avs_rdq_din;
logic avs_rdq_pop;
t_local_mem_data avs_rdq_dout;
logic avs_rdq_empty;
logic avs_rdq_full;

// CSR variables //////////////////////////////////////////////////////////////

logic [2:0]       csr_cmd;
t_ccip_clAddr     csr_io_addr;
t_local_mem_addr  csr_mem_addr;
logic [DRAM_ADDR_WIDTH-1:0] csr_data_size;

// MMIO controller ////////////////////////////////////////////////////////////

t_ccip_c0_ReqMmioHdr mmioHdr;

always_comb 
begin
  mmioHdr = t_ccip_c0_ReqMmioHdr'(cp2af_sRxPort.c0.hdr);
end

always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    af2cp_sTxPort.c2.hdr         <= 0;
    af2cp_sTxPort.c2.data        <= 0;
    af2cp_sTxPort.c2.mmioRdValid <= 0;
    csr_cmd                      <= 0;
    csr_io_addr                  <= 0;
    csr_mem_addr                 <= 0;
    csr_data_size                <= 0;
  end
  else begin

    csr_cmd <= 0;
    af2cp_sTxPort.c2.mmioRdValid <= 0;

    // serve MMIO write request
    if (cp2af_sRxPort.c0.mmioWrValid)
    begin
      case (mmioHdr.address)
        MMIO_CSR_IO_ADDR: begin                     
          csr_io_addr <= t_ccip_clAddr'(cp2af_sRxPort.c0.data);          
          $display("%t: CSR_IO_ADDR: 0x%h", $time, t_ccip_clAddr'(cp2af_sRxPort.c0.data));
        end
        MMIO_CSR_MEM_ADDR: begin          
          csr_mem_addr <= t_local_mem_addr'(cp2af_sRxPort.c0.data);                  
          $display("%t: CSR_MEM_ADDR: 0x%h", $time, t_local_mem_addr'(cp2af_sRxPort.c0.data));
        end
        MMIO_CSR_DATA_SIZE: begin          
          csr_data_size <= $bits(csr_data_size)'(cp2af_sRxPort.c0.data);          
          $display("%t: CSR_DATA_SIZE: %0d", $time, $bits(csr_data_size)'(cp2af_sRxPort.c0.data));
        end
        MMIO_CSR_CMD: begin          
          csr_cmd <= $bits(csr_cmd)'(cp2af_sRxPort.c0.data);
          $display("%t: CSR_CMD: %0d", $time, $bits(csr_cmd)'(cp2af_sRxPort.c0.data));
        end
        default: begin
           // user-defined CSRs
           //if (mmioHdr.addres >= MMIO_CSR_USER) begin
             // write Vortex CRS
           //end
        end 
      endcase
    end

    // serve MMIO read requests
    if (cp2af_sRxPort.c0.mmioRdValid) begin
      af2cp_sTxPort.c2.hdr.tid <= mmioHdr.tid; // copy TID
      case (mmioHdr.address)
        // AFU header
        16'h0000: af2cp_sTxPort.c2.data <= {
          4'b0001, // Feature type = AFU
          8'b0,    // reserved
          4'b0,    // afu minor revision = 0
          7'b0,    // reserved
          1'b1,    // end of DFH list = 1 
          24'b0,   // next DFH offset = 0
          4'b0,    // afu major revision = 0
          12'b0    // feature ID = 0
        };            
        AFU_ID_L: af2cp_sTxPort.c2.data <= afu_id[63:0];   // afu id low
        AFU_ID_H: af2cp_sTxPort.c2.data <= afu_id[127:64]; // afu id hi
        16'h0006: af2cp_sTxPort.c2.data <= 64'h0; // next AFU
        16'h0008: af2cp_sTxPort.c2.data <= 64'h0; // reserved
        MMIO_CSR_STATUS: begin
          if (state != af2cp_sTxPort.c2.data) begin
            $display("%t: STATUS: state=%0d", $time, state);
          end
          af2cp_sTxPort.c2.data <= state;
        end  
        default: af2cp_sTxPort.c2.data <= 64'h0;
      endcase
      af2cp_sTxPort.c2.mmioRdValid <= 1; // post response
    end
  end
end

// COMMAND FSM ////////////////////////////////////////////////////////////////

logic [DRAM_ADDR_WIDTH-1:0] cci_write_ctr;
logic [DRAM_ADDR_WIDTH-1:0] avs_read_ctr;
logic [DRAM_ADDR_WIDTH-1:0] avs_write_ctr;
logic                       vx_reset;

logic cmd_read_done;
logic cmd_write_done;
logic cmd_run_done;
logic cmd_clflush_done;

always_comb 
begin
  cmd_run_done = !vx_busy;
end

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
            $display("%t: STATE READ: ia=%h da=%h sz=%0d", $time, csr_io_addr, csr_mem_addr, csr_data_size);
            state <= STATE_READ;   
          end 
          CMD_TYPE_WRITE: begin      
            $display("%t: STATE WRITE: ia=%h da=%h sz=%0d", $time, csr_io_addr, csr_mem_addr, csr_data_size);
            state <= STATE_WRITE;
          end
          CMD_TYPE_RUN: begin        
            $display("%t: STATE START", $time);
            vx_reset <= 1;
            state <= STATE_START;                    
          end
          CMD_TYPE_CLFLUSH: begin
            $display("%t: STATE CFLUSH: da=%h sz=%0d", $time, csr_mem_addr, csr_data_size);
            state <= STATE_CLFLUSH;
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

    endcase
  end
end

// AVS Controller /////////////////////////////////////////////////////////////

logic vortex_enabled;
logic cci_rdq_empty;
t_cci_rdq_data cci_rdq_dout;
logic cci_rdq_pop;
logic cci_dram_req_read_fire;
logic cci_dram_req_write_fire;
logic vx_dram_req_read_fire;
logic vx_dram_req_write_fire;
logic [`LOG2UP(AVS_RD_QUEUE_SIZE):0] avs_pending_reads, avs_pending_rds_next;

t_ccip_clAddr next_avs_address;
always_comb 
begin
  vortex_enabled = (STATE_RUN == state) || (STATE_CLFLUSH == state);

  next_avs_address = csr_mem_addr + {avs_write_ctr[DRAM_ADDR_WIDTH-1:$bits(t_cci_rdq_tag)], t_cci_rdq_tag'(cci_rdq_dout)};

  cci_rdq_pop = (state == STATE_WRITE
              && !cci_rdq_empty 
              && !avs_waitrequest
              && avs_write_ctr < csr_data_size);

  cci_dram_req_read_fire = (state == STATE_READ) 
                        && (avs_pending_reads < AVS_RD_QUEUE_SIZE)
                        && !avs_waitrequest 
                        && avs_read_ctr < csr_data_size;

  cci_dram_req_write_fire = (state == STATE_WRITE) 
                         && cci_rdq_pop;

  vx_dram_req_read_fire  = vx_dram_req_read && vx_dram_req_ready;

  vx_dram_req_write_fire = vx_dram_req_write && vx_dram_req_ready;

  if ((cci_dram_req_read_fire || vx_dram_req_read_fire)
   && ~avs_readdatavalid) begin
    avs_pending_rds_next = avs_pending_reads + 1;
  end else 
  if (~(cci_dram_req_read_fire || vx_dram_req_read_fire)
   && avs_readdatavalid) begin
    avs_pending_rds_next = avs_pending_reads - 1;
  end else begin
    avs_pending_rds_next = avs_pending_reads;
  end

  cmd_write_done = (avs_write_ctr >= csr_data_size);
end

always_ff @(posedge clk) 
begin
  if (SoftReset) 
  begin    
    mem_bank_select   <= 0;
    avs_burstcount    <= 1;
    avs_byteenable    <= 64'hffffffffffffffff; 
    avs_read          <= 0;
    avs_write         <= 0;
    avs_read_ctr      <= 0;
    avs_write_ctr     <= 0;
    avs_pending_reads <= 0;
  end
  else begin

    avs_read <= 0;
    avs_write <= 0;

    if (state == STATE_IDLE) begin
      avs_read_ctr  <= 0;
      avs_write_ctr <= 0;
    end

    if (cci_dram_req_read_fire) begin
      avs_address  <= csr_mem_addr + avs_read_ctr;          
      avs_read_ctr <= avs_read_ctr + 1;          
      avs_read     <= 1;
      $display("%t: AVS Rd Req: addr=%h, pending=%0d", $time, (csr_mem_addr + avs_read_ctr), avs_pending_reads);
    end

    if (cci_dram_req_write_fire) begin          
      avs_writedata <= cci_rdq_dout[$bits(t_ccip_clData) + $bits(t_cci_rdq_tag)-1:$bits(t_cci_rdq_tag)];
      avs_address   <= next_avs_address;          
      avs_write_ctr <= avs_write_ctr + 1;
      avs_write     <= 1;
      $display("%t: AVS Wr Req: addr=%h (%0d/%0d)", $time, next_avs_address, avs_write_ctr + 1, csr_data_size);
    end

    if (vx_dram_req_read_fire) begin
      avs_address <= vx_dram_req_addr;
      avs_read    <= 1;
      $display("%t: AVS Rd Req: addr=%h, pending=%0d", $time, vx_dram_req_addr, avs_pending_reads);
    end 
    
    if (vx_dram_req_write_fire) begin
      avs_address   <= vx_dram_req_addr;
      avs_writedata <= vx_dram_req_data;          
      avs_write     <= 1;
      $display("%t: AVS Wr Req: addr=%h", $time, vx_dram_req_addr);
    end   

    if (avs_readdatavalid) begin
      $display("%t: AVS Rd Rsp: pending=%0d", $time, avs_pending_rds_next);
    end

    avs_pending_reads <= avs_pending_rds_next;   
  end
end

// Vortex DRAM requests

always_comb 
begin  
  vx_dram_req_ready = vortex_enabled && !avs_waitrequest && (avs_pending_reads < AVS_RD_QUEUE_SIZE);
end

// Vortex DRAM fill response

always_comb 
begin
  vx_dram_rsp_valid = vortex_enabled && !avs_rdq_empty;
  vx_dram_rsp_tag   = avs_rtq_dout;
  vx_dram_rsp_data  = avs_rdq_dout;
end

// AVS address read request queue /////////////////////////////////////////////

logic cci_wr_req;

always_comb 
begin
  avs_rtq_push = vx_dram_req_read_fire;
  avs_rtq_din  = vx_dram_req_tag;
  avs_rtq_pop  = vx_dram_rsp_valid;
end

VX_generic_queue #(
  .DATAW(DRAM_TAG_WIDTH),
  .SIZE(AVS_RD_QUEUE_SIZE)
) avs_rd_req_queue (
  .clk      (clk),
  .reset    (SoftReset),
  .push     (avs_rtq_push),
  .data_in  (avs_rtq_din),
  .pop      (avs_rtq_pop),
  .data_out (avs_rtq_dout),
  .empty    (avs_rtq_empty),
  .full     (avs_rtq_full)
);

// AVS data read response queue ///////////////////////////////////////////////

always_comb 
begin
  avs_rdq_push = avs_readdatavalid;
  avs_rdq_din  = avs_readdata;
  avs_rdq_pop  = vx_dram_rsp_valid || cci_wr_req; 
end

VX_generic_queue #(
  .DATAW(DRAM_LINE_WIDTH),
  .SIZE(AVS_RD_QUEUE_SIZE)
) avs_rd_rsp_queue (
  .clk      (clk),
  .reset    (SoftReset),
  .push     (avs_rdq_push),
  .data_in  (avs_rdq_din),
  .pop      (avs_rdq_pop),
  .data_out (avs_rdq_dout),
  .empty    (avs_rdq_empty),
  .full     (avs_rdq_full)
);

// CCI Read Request ///////////////////////////////////////////////////////////

t_ccip_c0_ReqMemHdr cci_read_hdr;

logic [DRAM_ADDR_WIDTH-1:0] cci_read_ctr;
t_cci_rdq_tag cci_rdq_ctr;

logic cci_rdq_full;
logic cci_rdq_push;
t_cci_rdq_data cci_rdq_din;

logic cci_read_wait;

always_comb 
begin
  cci_read_hdr = t_ccip_c0_ReqMemHdr'(0);
  cci_read_hdr.address = csr_io_addr + cci_read_ctr;  
  cci_read_hdr.mdata = t_cci_rdq_tag'(cci_read_ctr);

  cci_rdq_push = (STATE_WRITE == state) && cp2af_sRxPort.c0.rspValid;
  cci_rdq_din  = {cp2af_sRxPort.c0.data, t_cci_rdq_tag'(cp2af_sRxPort.c0.hdr.mdata)};  
end

// Send read requests to CCI
always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    af2cp_sTxPort.c0.hdr   <= 0;
    af2cp_sTxPort.c0.valid <= 0;
    cci_read_ctr           <= 0;
    cci_rdq_ctr            <= 0;
    cci_read_wait          <= 0;
  end 
  else begin      
    af2cp_sTxPort.c0.valid <= 0;

    if (STATE_IDLE == state) begin
      cci_read_ctr  <= 0;
      cci_rdq_ctr   <= 0;
      cci_read_wait <= 0;
    end

    if (STATE_WRITE == state 
     && !cp2af_sRxPort.c0TxAlmFull      // ensure read queue not full
     && !cci_rdq_full                   // ensure destination queue not full
     && !cci_read_wait                  // ensure the last batch has arrived
     && cci_read_ctr < csr_data_size)   // ensure not done
    begin
      af2cp_sTxPort.c0.hdr   <= cci_read_hdr;
      af2cp_sTxPort.c0.valid <= 1;      
      cci_read_ctr           <= cci_read_ctr + 1;
      if (t_cci_rdq_tag'(cci_read_ctr) == (CCI_RD_WINDOW_SIZE-1)) begin
        cci_read_wait <= 1;             // end current request batch
      end 
      $display("%t: CCI Rd Req: addr=%h, ctr=%0d", $time, cci_read_hdr.address, cci_read_ctr);
    end

    if (cci_rdq_push) begin
      cci_rdq_ctr <= cci_rdq_ctr + 1;
      if (cci_rdq_ctr == (CCI_RD_WINDOW_SIZE-1)) begin
        cci_read_wait <= 0;             // restart new request batch
      end 
      $display("%t: CCI Rd Rsp: idx=%0d, ctr=%0d", $time, t_cci_rdq_tag'(cp2af_sRxPort.c0.hdr.mdata), cci_rdq_ctr);
    end        
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
  .full     (cci_rdq_full)
);

// CCI Write Request //////////////////////////////////////////////////////////

t_ccip_c1_ReqMemHdr cci_write_hdr;

logic [DRAM_ADDR_WIDTH:0] cci_pending_writes, cci_pending_writes_next;

always_comb 
begin
  cci_wr_req = (STATE_READ == state) 
            && !avs_rdq_empty 
            && !cp2af_sRxPort.c1TxAlmFull
            && (cci_write_ctr < csr_data_size);  

  if (cci_wr_req && ~cp2af_sRxPort.c1.rspValid) begin
    cci_pending_writes_next = cci_pending_writes + 1;
  end else
  if (~cci_wr_req && cp2af_sRxPort.c1.rspValid) begin
    cci_pending_writes_next = cci_pending_writes - 1;
  end else begin
    cci_pending_writes_next = cci_pending_writes;
  end

  cci_write_hdr = t_ccip_c1_ReqMemHdr'(0);
  cci_write_hdr.address = csr_io_addr + cci_write_ctr;
  cci_write_hdr.sop = 1; // single line write mode

  cmd_read_done = (cci_write_ctr >= csr_data_size) && (0 == cci_pending_writes);
end

// Send write requests to CCI
always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    af2cp_sTxPort.c1.hdr   <= 0;
    af2cp_sTxPort.c1.data  <= 0;
    af2cp_sTxPort.c1.valid <= 0;
    cci_write_ctr          <= 0;
    cci_pending_writes     <= 0;
  end
  else begin
    af2cp_sTxPort.c1.valid <= 0;

    if (STATE_IDLE == state) begin
      cci_write_ctr <= 0;
    end

    if (cci_wr_req) begin
      af2cp_sTxPort.c1.hdr   <= cci_write_hdr;
      af2cp_sTxPort.c1.data  <= t_ccip_clData'(avs_rdq_dout);
      af2cp_sTxPort.c1.valid <= 1;            
      cci_write_ctr          <= cci_write_ctr + 1;
      $display("%t: CCI Wr Req: addr=%h (%0d/%0d)", $time, cci_write_hdr.address, cci_write_ctr + 1, csr_data_size);
    end

    if (cp2af_sRxPort.c1.rspValid) begin      
      $display("%t: CCI Wr Rsp: pending=%0d", $time, cci_pending_writes_next);      
    end

    cci_pending_writes <= cci_pending_writes_next;
  end
end

// Vortex cache snooping //////////////////////////////////////////////////////

logic [DRAM_ADDR_WIDTH-1:0] snp_req_ctr;
logic [DRAM_ADDR_WIDTH-1:0] snp_rsp_ctr;

always_comb 
begin
  cmd_clflush_done = (snp_rsp_ctr >= csr_data_size);
end

always_ff @(posedge clk) 
begin
  if (SoftReset) begin
    vx_snp_req_valid <= 0;
    vx_snp_req_tag   <= 0;
    vx_snp_rsp_ready <= 0;
    snp_req_ctr      <= 0;
    snp_rsp_ctr      <= 0;
  end
  else begin
    if (STATE_IDLE == state) begin
      snp_req_ctr      <= 0;
      snp_rsp_ctr      <= 0;
      vx_snp_rsp_ready <= 0;
    end

    vx_snp_req_valid <= 0;

    if ((STATE_CLFLUSH == state)
     && (snp_req_ctr < csr_data_size)
     && vx_snp_req_ready)
    begin
      vx_snp_req_addr  <= csr_mem_addr + snp_req_ctr;
      snp_req_ctr      <= snp_req_ctr + 1;
      vx_snp_req_valid <= 1;
      vx_snp_rsp_ready <= 1;
    end

    if ((STATE_CLFLUSH == state) 
     && (snp_rsp_ctr < csr_data_size)
     && vx_snp_rsp_valid
     && vx_snp_rsp_ready) begin
       snp_rsp_ctr <= snp_rsp_ctr + 1;
    end   
  end
end

// Vortex binding /////////////////////////////////////////////////////////////

Vortex_Socket #() vx_socket (
  .clk              (clk),
  .reset            (vx_reset),

  // DRAM request 
  .dram_req_write   (vx_dram_req_write),
  .dram_req_read 	  (vx_dram_req_read),
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
  .snp_req_tag      (vx_snp_req_tag),
  .snp_req_ready    (vx_snp_req_ready),

  // Snoop response
  .snp_rsp_valid 	  (vx_snp_rsp_valid),
  .snp_rsp_tag      (vx_snp_rsp_tag),
  .snp_rsp_ready    (vx_snp_rsp_ready),

  // I/O request
  .io_req_read      (),
  .io_req_write     (),    
  .io_req_addr      (),
  .io_req_data      (),
  .io_req_byteen    (),
  .io_req_tag       (),    
  .io_req_ready     (1'b1),

  // I/O response
  .io_rsp_valid     (1'b0),
  .io_rsp_data      (32'b0),
  .io_rsp_tag       (`CORE_REQ_TAG_WIDTH'(0)),
  .io_rsp_ready     (),
 
  // status
  .busy 				    (vx_busy),
  .ebreak           ()
);

endmodule
