`include "VX_define.v"
`include "VX_d_cache.v"

module VX_d_cache_tb;

  parameter NUMBER_BANKS = 8;

  reg clk, reset, im_ready;
  reg [`NT_M1:0] i_p_valid;
  reg  [`NT_M1:0][13:0] i_p_addr; // FIXME
  reg        i_p_initial_request;
  reg [`NT_M1:0][31:0]  i_p_writedata;
  reg         i_p_read_or_write; //, i_p_write;
  reg [`NT_M1:0][31:0]  o_p_readdata;
  reg [`NT_M1:0]        o_p_readdata_valid;
  reg        o_p_waitrequest;
  reg [13:0]  o_m_addr; // Only one address is sent out at a time to memory
  reg         o_m_valid;
  reg [(NUMBER_BANKS * 32) - 1:0] o_m_writedata;
  reg         o_m_read_or_write; //, o_m_write;
  reg [(NUMBER_BANKS * 32) - 1:0] i_m_readdata;  // Read Data that is passed from the memory module back to the controller


  VX_d_cache d_cache(.clk(clk),
               .rst(reset),
               .i_p_initial_request(i_p_initial_request),
               .i_p_addr(i_p_addr),
               .i_p_writedata(i_p_writedata),
               .i_p_read_or_write(i_p_read_or_write), // 0 = Read | 1 = Write
               .i_p_valid(i_p_valid),
               .o_p_readdata(o_p_readdata),
               .o_p_readdata_valid(o_p_readdata_valid),
               .o_p_waitrequest(o_p_waitrequest), // 0 = all threads done | 1 = Still threads that need to 
               .o_m_addr(o_m_addr),
               .o_m_writedata(o_m_writedata),
               .o_m_read_or_write(o_m_read_or_write), // 0 = Read | 1 = Write
               .o_m_valid(o_m_valid),
               .i_m_readdata(i_m_readdata),
               .i_m_ready(im_ready)
               //cnt_r,
               //cnt_w,
               //cnt_hit_r,
               //cnt_hit_w
  );



  initial 
  begin 
    clk = 0; 
    reset = 0; 
    
  end

  always 
    #5  clk =  ! clk; 

endmodule