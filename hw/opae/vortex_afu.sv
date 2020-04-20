// Interface between CSR and FSM
// All the MMIOs read/write are done from CSR and passed to the FSM for state transitions

// To be done:
// Change address size to buffer's address size and data size based on IO address size. Check from hello_world

`include "platform_if.vh"
import local_mem_cfg_pkg::*;
`include "afu_json_info.vh"

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

localparam AVS_RD_QUEUE_SIZE  = 16;

localparam VX_SNOOP_DELAY     = 300;
localparam VX_SNOOP_LEVELS    = 2;

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
  STATE_RUN, 
  STATE_CLFLUSH
} state_t;

state_t state;

// Vortex signals /////////////////////////////////////////////////////////////

logic        vx_dram_req_read;
logic        vx_dram_req_write;
logic [31:0] vx_dram_req_addr;
logic [31:0] vx_dram_req_data[15:0];
logic        vx_dram_req_full;

logic        vx_dram_rsp_ready;
logic        vx_dram_rsp_valid;
logic [31:0] vx_dram_rsp_addr;
logic [31:0] vx_dram_rsp_data[15:0];

logic        vx_snp_req;
logic [31:0] vx_snp_req_addr;
logic        vx_snp_req_full;

logic        vx_ebreak;

// AVS Queues /////////////////////////////////////////////////////////////////

logic avs_raq_push;
t_local_mem_addr avs_raq_din;
logic avs_raq_pop;
t_local_mem_addr avs_raq_dout;
logic avs_raq_empty;
logic avs_raq_full;

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
logic [31:0]      csr_data_size;

// MMIO controller ////////////////////////////////////////////////////////////

t_ccip_c0_ReqMmioHdr mmioHdr;

always_comb 
begin
  mmioHdr = t_ccip_c0_ReqMmioHdr'(cp2af_sRxPort.c0.hdr);
end

always_ff @(posedge clk) 
begin
  if (SoftReset) 
  begin
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
          csr_io_addr <= t_ccip_clAddr'(cp2af_sRxPort.c0.data >> 6);          
          $display("%t: CSR_IO_ADDR: 0x%h", $time, t_ccip_clAddr'(cp2af_sRxPort.c0.data >> 6));
        end
        MMIO_CSR_MEM_ADDR: begin          
          csr_mem_addr <= t_local_mem_addr'(cp2af_sRxPort.c0.data >> 6);                  
          $display("%t: CSR_MEM_ADDR: 0x%h", $time, t_local_mem_addr'(cp2af_sRxPort.c0.data >> 6));
        end
        MMIO_CSR_DATA_SIZE: begin          
          csr_data_size <= $bits(csr_data_size)'((cp2af_sRxPort.c0.data + 63) >> 6);          
          $display("%t: CSR_DATA_SIZE: %0d", $time, $bits(csr_data_size)'((cp2af_sRxPort.c0.data + 63) >> 6));
        end
        MMIO_CSR_CMD: begin          
          csr_cmd <= $bits(csr_cmd)'(cp2af_sRxPort.c0.data);
          $display("%t: CSR_CMD: %0d", $time, $bits(csr_cmd)'(cp2af_sRxPort.c0.data));
        end
      endcase
    end

    // serve MMIO read requests
    if (cp2af_sRxPort.c0.mmioRdValid)
    begin
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
          if (state != af2cp_sTxPort.c2.data) 
            $display("%t: STATUS: state=%0d", $time, state);
          af2cp_sTxPort.c2.data <= state;
        end  
        default: af2cp_sTxPort.c2.data <= 64'h0;
      endcase
      af2cp_sTxPort.c2.mmioRdValid <= 1; // post response
    end
  end
end

// COMMAND FSM ////////////////////////////////////////////////////////////////

logic [31:0] cci_write_ctr;
logic [31:0] avs_read_ctr;
logic [31:0] avs_write_ctr;
logic [31:0] vx_snoop_ctr;
logic [9:0]  vx_snoop_delay;
logic        vx_reset;

always_ff @(posedge clk) 
begin
  if (SoftReset) 
  begin
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
            state <= STATE_RUN;                    
          end
          CMD_TYPE_CLFLUSH: begin
            $display("%t: STATE CFLUSH: da=%h sz=%0d", $time, csr_mem_addr, csr_data_size);
            state <= STATE_CLFLUSH;
          end
        endcase
      end      

      STATE_READ: begin
        if (cci_write_ctr >= csr_data_size) 
        begin
          state <= STATE_IDLE;
        end
      end

      STATE_WRITE: begin
        if (avs_write_ctr >= csr_data_size) 
        begin
          state <= STATE_IDLE;
        end
      end

      STATE_RUN: begin
        if (vx_ebreak)
        begin
          state <= STATE_IDLE;
        end
      end

      STATE_CLFLUSH: begin
        if (vx_snoop_delay >= VX_SNOOP_DELAY) 
        begin
          state <= STATE_IDLE;
        end
      end

    endcase
  end
end

// AVS Controller /////////////////////////////////////////////////////////////

always_ff @(posedge clk) 
begin
  if (SoftReset) 
  begin    
    mem_bank_select <= 0;
    avs_burstcount  <= 1;
    avs_byteenable  <= 64'hffffffffffffffff;    
    avs_address     <= 0;
    avs_writedata   <= 0;    
    avs_read        <= 0;
    avs_write       <= 0;
    avs_read_ctr    <= 0;
    avs_write_ctr   <= 0;
  end
  else begin

    avs_read <= 0;
    avs_write <= 0;

    case (state)
      STATE_IDLE: begin
        avs_read_ctr <= 0;
        avs_write_ctr <= 0;
      end

      STATE_READ: begin
        if (!avs_raq_full 
         && !avs_rdq_full
         && !avs_waitrequest 
         && avs_read_ctr < csr_data_size) 
        begin
          avs_address <= csr_mem_addr + avs_read_ctr;
          avs_read <= 1;
          avs_read_ctr <= avs_read_ctr + 1;
          $display("%t: AVS Rd Req: addr=%h", $time, csr_mem_addr + avs_read_ctr);
        end        
      end

      STATE_WRITE: begin
        if (cp2af_sRxPort.c0.rspValid 
         && avs_write_ctr < csr_data_size) 
        begin
          avs_writedata <= cp2af_sRxPort.c0.data;
          avs_address <= csr_mem_addr + avs_write_ctr;
          avs_write <= 1;
          avs_write_ctr <= avs_write_ctr + 1;
          $display("%t: AVS Wr Req: addr=%h (%0d/%0d)", $time, csr_mem_addr + avs_write_ctr, avs_write_ctr + 1, csr_data_size);
        end
      end

      STATE_RUN, STATE_CLFLUSH: begin
        if (vx_dram_req_read
         && !vx_dram_req_full) 
        begin
          avs_address <= (vx_dram_req_addr >> 6);
          avs_read <= 1;
          $display("%t: AVS Rd Req: addr=%h", $time, vx_dram_req_addr >> 6);
        end 
        
        if (vx_dram_req_write
         && !vx_dram_req_full) 
        begin
          avs_writedata <= {>>{vx_dram_req_data}};
          avs_address <= (vx_dram_req_addr >> 6);
          avs_write <= 1;
          $display("%t: AVS Wr Req: addr=%h", $time, vx_dram_req_addr >> 6);
        end
      end
    endcase

    if (avs_readdatavalid) 
    begin
      $display("%t: AVS Rd Rsp", $time);
    end   
  end
end

// Vortex DRAM requests stalling

logic vortex_enabled;

always_comb 
begin
  vortex_enabled    = (STATE_RUN == state) || (STATE_CLFLUSH == state);
  vx_dram_req_ready = vortex_enabled && !avs_waitrequest && !avs_raq_full && !avs_rdq_full;
end

// Vortex DRAM fill response

always_comb 
begin
  vx_dram_rsp_valid      = vortex_enabled && !avs_rdq_empty && vx_dram_rsp_ready;
  vx_dram_rsp_addr       = (avs_raq_dout << 6);
  {>>{vx_dram_rsp_data}} = avs_rdq_dout;
end

// AVS address read request queue /////////////////////////////////////////////

logic cci_write_req;

always_comb 
begin
  avs_raq_pop  = vx_dram_rsp_valid || cci_write_req;
  avs_raq_din  = avs_address;
  avs_raq_push = avs_read;
end

VX_generic_queue #(
  .DATAW($bits(t_local_mem_addr)),
  .SIZE(AVS_RD_QUEUE_SIZE)
) vx_rd_addr_queue (
  .clk      (clk),
  .reset    (SoftReset),
  .push     (avs_raq_push),
  .in_data  (avs_raq_din),
  .pop      (avs_raq_pop),
  .out_data (avs_raq_dout),
  .empty    (avs_raq_empty),
  .full     (avs_raq_full)
);

// AVS data read response queue ///////////////////////////////////////////////

always_comb 
begin
  avs_rdq_pop  = avs_raq_pop;
  avs_rdq_din  = avs_readdata;
  avs_rdq_push = avs_readdatavalid;
end

VX_generic_queue #(
  .DATAW($bits(t_local_mem_data)),
  .SIZE(AVS_RD_QUEUE_SIZE)
) vx_rd_data_queue (
  .clk      (clk),
  .reset    (SoftReset),
  .push     (avs_rdq_push),
  .in_data  (avs_rdq_din),
  .pop      (avs_rdq_pop),
  .out_data (avs_rdq_dout),
  .empty    (avs_rdq_empty),
  .full     (avs_rdq_full)
);

// CCI Read Request ///////////////////////////////////////////////////////////

t_ccip_c0_ReqMemHdr rd_hdr;

logic cci_read_pending;

always_comb 
begin
  rd_hdr = t_ccip_c0_ReqMemHdr'(0);
  rd_hdr.address = csr_io_addr + avs_write_ctr;  
end

// Send read requests to CCI
always_ff @(posedge clk) 
begin
  if (SoftReset) 
  begin
    af2cp_sTxPort.c0.hdr   <= 0;
    af2cp_sTxPort.c0.valid <= 0;
    cci_read_pending       <= 0;
  end 
  else begin      
    af2cp_sTxPort.c0.valid <= 0;

    if (STATE_WRITE == state 
     && !cp2af_sRxPort.c0TxAlmFull      // ensure read queue not full
     && !avs_waitrequest                // ensure AVS write queue not full
     && !cci_read_pending               // ensure no read pending
     && avs_write_ctr < csr_data_size)  // ensure not done
    begin
      af2cp_sTxPort.c0.hdr <= rd_hdr;
      af2cp_sTxPort.c0.valid <= 1;      
      cci_read_pending <= 1;
      $display("%t: CCI Rd Req: addr=%h", $time, rd_hdr.address);
    end

    if (cci_read_pending
     && cp2af_sRxPort.c0.rspValid) 
    begin
      $display("%t: CCI Rd Rsp", $time);
      cci_read_pending <= 0;
    end    
  end
end

// CCI Write Request //////////////////////////////////////////////////////////

t_ccip_c1_ReqMemHdr wr_hdr;

logic cci_write_pending;

always_comb 
begin
  cci_write_req = (STATE_READ == state) 
               && !avs_rdq_empty 
               && !cp2af_sRxPort.c1TxAlmFull
               && !cci_write_pending
               && cci_write_ctr < csr_data_size;

  wr_hdr = t_ccip_c1_ReqMemHdr'(0);
  wr_hdr.address = csr_io_addr + cci_write_ctr;
  wr_hdr.sop = 1; // single line write mode
end

// Send write requests to CCI
always_ff @(posedge clk) 
begin
  if (SoftReset) 
  begin
    af2cp_sTxPort.c1.hdr   <= 0;
    af2cp_sTxPort.c1.data  <= 0;
    af2cp_sTxPort.c1.valid <= 0;
    cci_write_ctr          <= 0;
    cci_write_pending      <= 0;
  end
  else begin
    af2cp_sTxPort.c1.valid <= 0;

    if (STATE_IDLE == state) 
    begin
      cci_write_ctr <= 0;
    end

    if (cci_write_req)
    begin
      af2cp_sTxPort.c1.hdr   <= wr_hdr;
      af2cp_sTxPort.c1.data  <= t_ccip_clData'(avs_rdq_dout);
      af2cp_sTxPort.c1.valid <= 1;      
      cci_write_pending      <= 1;
      $display("%t: CCI Wr Req: addr=%h", $time, wr_hdr.address);
    end

    if (cci_write_pending
     && cp2af_sRxPort.c1.rspValid) 
    begin
      cci_write_ctr     <= cci_write_ctr + 1;
      cci_write_pending <= 0;
      $display("%t: CCI Wr Rsp (%0d/%0d)", $time, cci_write_ctr + 1, csr_data_size);      
    end
  end
end

// Vortex cache snooping //////////////////////////////////////////////////////

always_ff @(posedge clk) 
begin
  if (SoftReset) 
  begin
    vx_snp_req      <= 0;
    vx_snoop_ctr    <= 0;
    vx_snoop_delay  <= 0;
  end
  else begin
    if (STATE_IDLE == state) 
    begin
      vx_snoop_ctr   <= 0;
      vx_snoop_delay <= 0;
    end

    vx_snp_req <= 0;

    if ((STATE_CLFLUSH == state)
     && vx_snoop_ctr < csr_data_size
     && vx_snp_req_ready)
    begin
      vx_snp_req_addr <= (csr_mem_addr + vx_snoop_ctr) << 6;
      vx_snp_req      <= 1;
      vx_snoop_ctr    <= vx_snoop_ctr + 1;
    end

    if (vx_snoop_ctr == csr_data_size)
    begin 
      vx_snoop_delay <= vx_snoop_delay + 1;
    end
  end
end

// Vortex binding /////////////////////////////////////////////////////////////

Vortex_Socket #() vx_socket (
  .clk                (clk),
  .reset              (SoftReset || vx_reset),

  // DRAM Req 
  .dram_req_write 		(vx_dram_req_write),
  .dram_req_read 			(vx_dram_req_read),
  .dram_req_addr 			(vx_dram_req_addr),
  .dram_req_data			(vx_dram_req_data),
  .dram_req_ready     (vx_dram_req_ready),

  // DRAM Rsp
  .out_dram_rsp_ready (vx_dram_rsp_ready),
  .dram_rsp_valid 		(vx_dram_rsp_valid),
  .out_dram_rsp_addr  (vx_dram_rsp_addr),
  .out_dram_rsp_data	(vx_dram_rsp_data),

  // Cache Snooping Req
  .llc_snp_req_valid 	(vx_snp_req),
  .llc_snp_req_addr   (vx_snp_req_addr),
  .llc_snp_req_ready  (vx_snp_req_ready),
 
  // program exit signal
  .ebreak 				    (vx_ebreak)
);

endmodule
