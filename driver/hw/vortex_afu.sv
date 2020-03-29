// Code reused from Intel OPAE's 04_local_memory sample program with changes made to fit Vortex

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

  // Avalong signals for local memory access
  output  t_local_mem_data  avs_writedata,
  input   t_local_mem_data  avs_readdata,
  output  t_local_mem_addr  avs_address,
  input   logic             avs_waitrequest,
  output  logic             avs_write,
  output  logic             avs_read,
  output  t_local_mem_byte_mask  avs_byteenable,
  output  t_local_mem_burst_cnt  avs_burstcount,
  input                     avs_readdatavalid,

  output logic [$clog2(NUM_LOCAL_MEM_BANKS)-1:0] mem_bank_select
);

localparam AFU_ID_L           = 16'h0002;      // AFU ID Lower
localparam AFU_ID_H           = 16'h0004;      // AFU ID Higher 
localparam MEM_ADDRESS        = 16'h0040;      // AVMM Master Address
localparam MEM_BURSTCOUNT     = 16'h0042;      // AVMM Master Burst Count
localparam MEM_RDWR           = 16'h0044;      // AVMM Master Read/Write
localparam MEM_BANK_SELECT    = 16'h0064;      // Memory bank selection register
localparam READY_FOR_SW_CMD   = 16'h0066;      // "Ready for sw cmd" register. S/w must poll this register before issuing a read/write command to fsm
localparam MEM_BYTEENABLE     = 16'h0068;      // Test byteenable

// Added by Apurve to supporead and writeChange address size to buffer's address size
localparam DATA_SIZE          = 16'h0046;    // MMIO set by SW to denote the size od data to read/write
localparam BUFFER_IO_ADDRESS  = 16'h0048;    // MMIO set by SW to denote the buffer address space

logic [127:0] afu_id = `AFU_ACCEL_UUID;

// cast c0 header into ReqMmioHdr
t_ccip_c0_ReqMmioHdr mmioHdr;
assign mmioHdr = t_ccip_c0_ReqMmioHdr'(cp2af_sRxPort.c0.hdr);

logic [2:0]  mem_RDWR = '0;

//--
logic ready_for_sw_cmd;
logic run_vortex;

logic [15:0]          avm_data_size;
t_ccip_clAddr         avm_write_buffer_address;
t_ccip_clAddr         avm_read_buffer_address;
logic                 avm_read;
logic                 avm_write;
t_local_mem_addr      avm_address;
t_local_mem_burst_cnt avm_burstcount;
t_local_mem_byte_mask avm_byteenable;

// Vortex signals

logic       vx_reset;
logic       vx_dram_req;
logic       vx_dram_req_write;
logic       vx_dram_req_read;
logic       vx_ebreak;
logic [31:0] vx_dram_req_addr;
logic [31:0] vx_local_addr;
logic [31:0] vx_dram_req_size;
logic [31:0] vx_count;
logic        vx_dram_fill_rsp;

logic [31:0] vx_dram_req_data[15:0];
logic [31:0] vx_dram_fill_rsp_data[15:0];
logic        vx_dram_fill_accept;
logic [31:0] vx_dram_fill_rsp_addr;
logic [31:0] vx_dram_expected_lat;

//
// MMIO control threads
//
always@(posedge clk) begin
  if(SoftReset) begin
    af2cp_sTxPort.c2.hdr        <= '0;
    af2cp_sTxPort.c2.data       <= '0;
    af2cp_sTxPort.c2.mmioRdValid <= '0;
    avm_address     <= '0;
    avm_read        <= '0;
    avm_write       <= '0;
    avm_burstcount  <= 12'd1;
    mem_RDWR        <= '0;
    mem_bank_select <= 1'b1;

    // Change address size to buffer's address size
    avm_data_size             <= '0;
    avm_write_buffer_address  <= '0;
    avm_read_buffer_address   <= '0;
    run_vortex                <= '0;
  end
  else begin
      af2cp_sTxPort.c2.mmioRdValid <= 0;
      avm_read  <= mem_RDWR[0] &  mem_RDWR[1]; //[0] enable [1] 0-WR,1-RD
      avm_write <= mem_RDWR[0] & !mem_RDWR[1];

      // Added by Apurve. Run vortex whem RDWR is 7
      run_vortex <= mem_RDWR[0] & mem_RDWR[1] & mem_RDWR[2];

      // set the registers on MMIO write request
      // these are user-defined AFU registers at offset 0x40 and 0x41
      if(cp2af_sRxPort.c0.mmioWrValid == 1)
      begin
        case(mmioHdr.address)
          MEM_ADDRESS: avm_address <= t_local_mem_addr'(cp2af_sRxPort.c0.data);
          MEM_BURSTCOUNT: avm_burstcount <= cp2af_sRxPort.c0.data[11:0];
          MEM_RDWR: mem_RDWR <= cp2af_sRxPort.c0.data[2:0];
          MEM_BANK_SELECT: mem_bank_select <= $bits(mem_bank_select)'(cp2af_sRxPort.c0.data);
          // Added by Apurve to support read and write buffers. Change address size to buffer's address size
          DATA_SIZE:avm_data_size  <= cp2af_sRxPort.c0.data[15:0];

          BUFFER_IO_ADDRESS: begin
		 avm_write_buffer_address <= t_ccip_clAddr'(cp2af_sRxPort.c0.data);
		 avm_read_buffer_address <= t_ccip_clAddr'(cp2af_sRxPort.c0.data);
	  end
        endcase
      end

      // serve MMIO read requests
      if(cp2af_sRxPort.c0.mmioRdValid == 1)
      begin
        af2cp_sTxPort.c2.hdr.tid <= mmioHdr.tid; // copy TID
        case(mmioHdr.address)
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
          AFU_ID_L:             af2cp_sTxPort.c2.data <= afu_id[63:0];   // afu id low
          AFU_ID_H:             af2cp_sTxPort.c2.data <= afu_id[127:64]; // afu id hi
          16'h0006:             af2cp_sTxPort.c2.data <= 64'h0; // next AFU
          16'h0008:             af2cp_sTxPort.c2.data <= 64'h0; // reserved
          MEM_ADDRESS:          af2cp_sTxPort.c2.data <= 64'(avm_address);
          MEM_BURSTCOUNT:       af2cp_sTxPort.c2.data <= 64'(avm_burstcount);
          MEM_RDWR:             af2cp_sTxPort.c2.data <= {62'd0, mem_RDWR};
          READY_FOR_SW_CMD:     af2cp_sTxPort.c2.data <= ready_for_sw_cmd;
	        MEM_BANK_SELECT:      af2cp_sTxPort.c2.data <= 64'(mem_bank_select);
          default:              af2cp_sTxPort.c2.data <= 64'h0;
        endcase
        af2cp_sTxPort.c2.mmioRdValid <= 1; // post response
      end else
      begin
          if (avm_read | avm_write | run_vortex) mem_RDWR[0] <= 0;
      end 
    end
end





// FSM 

// Code reused from Intel OPAE's 04_local_memory sample program with changes made to fit Vortex

// Interface between CSR and FSM
// All the MMIOs read/write passed from csr are used for state transitions
// Read: local memory to shared buffer
// Write: shared buffer to local memory

// To be done:
// Review the FSM and implement read/write to shared buffer
// Vortex on/off signal
// check on byteenable and burst signals

//cp2af_sRxPort -> sRx
//af2cp_sTxPort -> sTx


typedef enum logic[3:0] { IDLE,
                          VX_REQ,
                          VX_WR_REQ,
                          VX_RD_REQ,
                          VX_RSP,
                          RD_REQ,
                          RD_RSP,
                          WR_REQ,
                          WR_RSP } state_t;


// Added by Apurve for shared memory space write/read
t_ccip_clAddr wr_addr;
t_ccip_clAddr rd_addr;
logic [15:0] count;
logic [15:0] count_rsp;
logic start_read;
logic start_write;
t_local_mem_addr local_address;
logic init_avs_read;

parameter ADDRESS_MAX_BIT = 10;
state_t state;

assign avs_burstcount = avm_burstcount;
t_local_mem_burst_cnt burstcount;

assign avs_byteenable = avm_byteenable;

always_ff @(posedge clk) begin
  if(SoftReset) begin
    local_address  <= '0;
    avs_write      <= '0;
    avs_read       <= '0;
    state          <= IDLE;
    burstcount     <= 1;
    ready_for_sw_cmd <= 0;
    count <= 0;
    count_rsp <= 0;
    vx_reset <= 1'b0;
    vx_count <= 0;
  end
  else begin
    case(state)
      IDLE: begin
        ready_for_sw_cmd <= 1;

        if (avm_write) begin
          state <= WR_REQ;
          ready_for_sw_cmd <= 0;
          count <= 0;
          count_rsp <= 0;
        end else if (avm_read) begin
	  init_avs_read <= 1;
          state <= RD_REQ;
          ready_for_sw_cmd <= 0;
          count <= 0;
          count_rsp <= 0;
        end else if (run_vortex) begin
          state <= VX_REQ;
          vx_reset <= 1'b1;
          ready_for_sw_cmd <= 0;
        end
      end

      WR_REQ: begin //AVL MM Posted Write
        af2cp_sTxPort.c0.valid <= 1'b0;
        avs_write <= 0;
        if (~avs_waitrequest)
        begin
          if (count_rsp >= avm_data_size)
          begin
            state <= WR_RSP;
            avs_write <= 0;
          end
        end
      end

      WR_RSP: begin // wait for write response
	avm_byteenable <= 64'hffffffffffffffff;
        state <= IDLE;
      end

      RD_REQ: begin // AVL MM Read non-posted
        af2cp_sTxPort.c1.valid <= 1'b0;
        if (~avs_waitrequest) begin
          if (count_rsp >= avm_data_size)
          begin
            state <= RD_RSP;
            avs_read <= 0;
          end
        end
      end

      RD_RSP: begin
        state <= IDLE;
      end

      VX_REQ: begin
        vx_reset <= 1'b0;
	if (vx_dram_req_write) begin
          vx_count <= 0;
          avs_write <= 1'b1;
          state <= VX_WR_REQ;
	end

	if (vx_dram_req_read) begin
          vx_count <= 0;
          avs_read <= 1'b1;
          state <= VX_RD_REQ;
	end

        if (vx_ebreak) begin
          state <= VX_RSP;
        end
      end

      VX_WR_REQ: begin
        avs_write <= 1'b0;
	if (vx_count >= vx_dram_req_size)
	begin
          state <= VX_REQ;
          vx_count <= 0;
        end
      end

      VX_RD_REQ: begin
        avs_read <= 1'b0;
	vx_dram_fill_rsp <= 1'b0;
	if (vx_count >= vx_dram_req_size)
	begin
          state <= VX_REQ;
          vx_count <= 0;
        end
      end

      VX_RSP: begin
        vx_count <= 0;
        state <= IDLE;
      end

    endcase
  end // end else reset
end // posedge clk


// Vortex call
  Vortex_SOC #() 
  vx_soc (
   .clk 					 (clk),
   .reset 					 (vx_reset),

    // IO
   //.io_valid[`NUMBER_CORES-1:0] 		 (),
   //.io_data [`NUMBER_CORES-1:0] 		 (),
   //.number_cores 				 (),

   // DRAM Dcache Req
   .out_dram_req 				 (vx_dram_req),
   .out_dram_req_write 				 (vx_dram_req_write),
   .out_dram_req_read 				 (vx_dram_req_read),
   .out_dram_req_addr 				 (vx_dram_req_addr),
   .out_dram_req_size 				 (vx_dram_req_size),
   .out_dram_req_data			 	 (vx_dram_req_data),
   .out_dram_expected_lat 			 (vx_dram_expected_lat),

   // DRAM Dcache Res
   .out_dram_fill_accept 			 (vx_dram_fill_accept),
   .out_dram_fill_rsp 				 (vx_dram_fill_rsp),
   .out_dram_fill_rsp_addr 			 (vx_dram_fill_rsp_addr),
   .out_dram_fill_rsp_data			 (vx_dram_fill_rsp_data),

   //.l3c_snp_req 				 (),
   //.l3c_snp_req_addr 				 (),
   //.l3c_snp_req_delay 			 (),

   .out_ebreak 					 (vx_ebreak)
  );


// Local memory read/write address
//assign avs_address = (vx_dram_req ? (vx_count ? vx_local_addr : vx_dram_req_addr) : (count ? local_address : avm_address));
assign avs_address = (((state == VX_WR_REQ) || (state == VX_RD_REQ)) ? (vx_count ? vx_local_addr : vx_dram_req_addr) : (count ? local_address : avm_address));



// Vortex DRAM requests and responses
// Handling of read/write data and vx_dram_req_size
// Is vx_dram_fill_accept for backpressure?
always_ff @(posedge clk) begin
  if (state == VX_WR_REQ) begin
    if (!avs_waitrequest & (vx_count < vx_dram_req_size)) begin
      avs_write <= 1'b1; 
      //avs_writedata <= vx_dram_req_data; 
      avs_writedata[31:0] = vx_dram_req_data[0];
      avs_writedata[63:32] = vx_dram_req_data[1];
      avs_writedata[95:64] = vx_dram_req_data[2];
      avs_writedata[127:96] = vx_dram_req_data[3];
      avs_writedata[159:128] = vx_dram_req_data[4];
      avs_writedata[191:160] = vx_dram_req_data[5];
      avs_writedata[223:192] = vx_dram_req_data[6];
      avs_writedata[255:224] = vx_dram_req_data[7];
      avs_writedata[287:256] = vx_dram_req_data[8];
      avs_writedata[319:288] = vx_dram_req_data[9];
      avs_writedata[351:320] = vx_dram_req_data[10];
      avs_writedata[383:352] = vx_dram_req_data[11];
      avs_writedata[415:384] = vx_dram_req_data[12];
      avs_writedata[447:416] = vx_dram_req_data[13];
      avs_writedata[479:448] = vx_dram_req_data[14];
      avs_writedata[511:480] = vx_dram_req_data[15];

      vx_local_addr <= (vx_count ? vx_local_addr + 1 : vx_dram_req_addr + 1);

      // Update the count value based on the number of bytes written
      vx_count <= vx_count + 64;

      if ((vx_dram_req_size - vx_count) < 64)
      begin
        avm_byteenable <= 64'hffffffffffffffff >> (64 - (vx_dram_req_size - vx_count));
      end else
      begin
        avm_byteenable <= 64'hffffffffffffffff;
      end

    end
  end
end

always_ff @(posedge clk) begin
  //if (SoftReset) begin
  if (vx_reset) begin
    vx_dram_fill_rsp <= 1'b0;
    //vx_dram_fill_rsp_data <= 0;
    vx_dram_fill_rsp_data[0] <= 0;
    vx_dram_fill_rsp_data[1] <= 0;
    vx_dram_fill_rsp_data[2] <= 0;
    vx_dram_fill_rsp_data[3] <= 0;
    vx_dram_fill_rsp_data[4] <= 0;
    vx_dram_fill_rsp_data[5] <= 0;
    vx_dram_fill_rsp_data[6] <= 0;
    vx_dram_fill_rsp_data[7] <= 0;
    vx_dram_fill_rsp_data[8] <= 0;
    vx_dram_fill_rsp_data[9] <= 0;
    vx_dram_fill_rsp_data[10] <= 0;
    vx_dram_fill_rsp_data[11] <= 0;
    vx_dram_fill_rsp_data[12] <= 0;
    vx_dram_fill_rsp_data[13] <= 0;
    vx_dram_fill_rsp_data[14] <= 0;
    vx_dram_fill_rsp_data[15] <= 0;
  end

  if (state == VX_RD_REQ) begin
    if (avs_readdatavalid & vx_dram_fill_accept) begin
      avs_read <= 1'b1;
      vx_dram_fill_rsp <= 1'b1;
      //vx_dram_fill_rsp_data <= avs_readdata;
      vx_dram_fill_rsp_data[0] <= avs_readdata[31:0];
      vx_dram_fill_rsp_data[1] <= avs_readdata[63:32];
      vx_dram_fill_rsp_data[2] <= avs_readdata[95:64];
      vx_dram_fill_rsp_data[3] <= avs_readdata[127:96];
      vx_dram_fill_rsp_data[4] <= avs_readdata[159:128];
      vx_dram_fill_rsp_data[5] <= avs_readdata[191:160];
      vx_dram_fill_rsp_data[6] <= avs_readdata[223:192];
      vx_dram_fill_rsp_data[7] <= avs_readdata[255:224];
      vx_dram_fill_rsp_data[8] <= avs_readdata[287:256];
      vx_dram_fill_rsp_data[9] <= avs_readdata[319:288];
      vx_dram_fill_rsp_data[10] <= avs_readdata[351:320];
      vx_dram_fill_rsp_data[11] <= avs_readdata[383:352];
      vx_dram_fill_rsp_data[12] <= avs_readdata[415:384];
      vx_dram_fill_rsp_data[13] <= avs_readdata[447:416];
      vx_dram_fill_rsp_data[14] <= avs_readdata[479:448];
      vx_dram_fill_rsp_data[15] <= avs_readdata[511:480];
      vx_local_addr <= (vx_count ? vx_local_addr + 1 : vx_dram_req_addr + 1);
      vx_dram_fill_rsp_addr <= vx_local_addr; 
      // Update the count value based on the number of bytes written
      vx_count <= vx_count + 64;

    end
  end
end




// Read from local memory (avs_readdata) and write to shared space
// Implement write header
always_ff @(posedge clk) begin
  if (state == RD_REQ & avs_readdatavalid & !cp2af_sRxPort.c1TxAlmFull & count < avm_data_size & !avs_waitrequest & start_write)
  begin
    wr_addr <= (count? wr_addr + 1 : avm_write_buffer_address + 1);
    local_address <= (count? local_address + 1 : avm_address + 1);
    start_write <= 1'b0;
  end
end

// Write header defines the request to the FIU
t_ccip_c1_ReqMemHdr wr_hdr;

always_comb
begin
  wr_hdr = t_ccip_c1_ReqMemHdr'(0);

  // Virtual address (MPF virtual addressing is enabled)
  wr_hdr.address = (count? wr_addr: avm_write_buffer_address);

  // Start of packet is true (single line write)
  wr_hdr.sop = 1'b1;
end

// Send write requests to the FIU
always_ff @(posedge clk)
begin
  if (SoftReset)
  begin
    af2cp_sTxPort.c1.hdr        <= '0;
    af2cp_sTxPort.c1.data       <= '0;
    af2cp_sTxPort.c1.valid      <= '0;
  end

  // Generate a write request when needed and the FIU isn't full
  if (state == RD_REQ & avs_readdatavalid & !cp2af_sRxPort.c1TxAlmFull & count < avm_data_size & !avs_waitrequest & start_write)
  begin
    af2cp_sTxPort.c1.hdr <= wr_hdr;
    af2cp_sTxPort.c1.data <= t_ccip_clData'(avs_readdata);
    af2cp_sTxPort.c1.valid <= 1'b1;
    start_write <= 1'b0;
    count <= count + 64;
  end
end

// Write response
always_ff @(posedge clk)
begin
  if (SoftReset)
  begin
    start_write <= 1'b1;
  end

  // Generate a read request when needed and the FIU isn't full
  if (state == RD_REQ & cp2af_sRxPort.c1.rspValid)
  begin
    count_rsp <= count_rsp + 64;
    start_write <= 1'b1;
    init_avs_read <= 1'b1;
  end
end


// avs_read control 

always_ff @(posedge clk)
begin
  if (SoftReset)
  begin
    init_avs_read <= 1'b0;
  end

  if (init_avs_read & state <= RD_REQ) 
  begin
    avs_read <= 1'b1;
    init_avs_read <= 1'b0;
  end else
  begin
    avs_read <= 1'b0;
  end
end




// Write to local memory (avs_writedata) and read from shared space
// Implement read header
always_ff @(posedge clk) begin
  if (SoftReset)
  begin
    rd_addr <= 0;
    local_address  <= 0; 
  end

  if (state == WR_REQ & !cp2af_sRxPort.c0TxAlmFull & count < avm_data_size & !avs_waitrequest & start_read)
  begin
    // Read address + 1 gives address for next block. Each block is 64B
    rd_addr <= (count? rd_addr + 1 : avm_read_buffer_address + 1);
    local_address <= (count? local_address + 1 : avm_address);
    start_read <= 1'b0;
  end
end

// Read header defines the request to the FIU
t_ccip_c0_ReqMemHdr rd_hdr;

always_comb
begin
  rd_hdr = t_ccip_c0_ReqMemHdr'(0);
  rd_hdr.address = (count? rd_addr : avm_read_buffer_address);  
end

// Send read requests to the FIU
always_ff @(posedge clk)
begin
  if (SoftReset)
  begin
    af2cp_sTxPort.c0.hdr        <= '0;
    af2cp_sTxPort.c0.valid      <= '0;
  end

  // Generate a read request when needed and the FIU isn't full
  if (state == WR_REQ & !cp2af_sRxPort.c0TxAlmFull & count < avm_data_size & !avs_waitrequest & start_read)
  begin
    af2cp_sTxPort.c0.hdr <= rd_hdr;
    af2cp_sTxPort.c0.valid <= 1'b1;
    start_read <= 1'b0;
    count <= count + 64;
  end
end

// Read response
always_ff @(posedge clk)
begin
  if (SoftReset)
  begin
    start_read    <= 1'b1;
    avm_byteenable <= 64'hffffffffffffffff;
  end

  // Generate a read request when needed and the FIU isn't full
  if (state == WR_REQ & cp2af_sRxPort.c0.rspValid)
  begin
    if ((avm_data_size - count_rsp) < 64)
    begin
      avm_byteenable <= 64'hffffffffffffffff >> (64 - (avm_data_size - count_rsp));
    end else
    begin
      avm_byteenable <= 64'hffffffffffffffff;
    end
    avs_writedata <= cp2af_sRxPort.c0.data;
    avs_write <= 1;
    count_rsp <= count_rsp + 64;
    start_read <= 1'b1;
  end
end

endmodule
