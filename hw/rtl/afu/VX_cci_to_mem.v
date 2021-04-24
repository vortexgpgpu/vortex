`include "VX_define.vh"

module VX_cci_to_mem #(
    parameter CCI_DATAW   = 1, 
    parameter CCI_ADDRW   = 1,            
    parameter AVS_DATAW   = 1, 
    parameter AVS_ADDRW   = 1,            
    parameter AVS_BYTEENW = (AVS_DATAW / 8),
    parameter TAG_WIDTH   = 1
) (
    input wire clk,
    input wire reset,

    input wire dram_req_valid_in,
    input wire [CCI_ADDRW-1:0] dram_req_addr_in,
    input wire dram_req_rw_in,
    input wire [CCI_DATAW-1:0] dram_req_data_in,
    input wire [TAG_WIDTH-1:0] dram_req_tag_in,
    output wire dram_req_ready_in,

    output wire dram_req_valid_out,
    output wire [AVS_ADDRW-1:0] dram_req_addr_out,
    output wire dram_req_rw_out,
    output wire [AVS_BYTEENW-1:0] dram_req_byteen_out,
    output wire [AVS_DATAW-1:0] dram_req_data_out,
    output wire [TAG_WIDTH-1:0] dram_req_tag_out,
    input wire dram_req_ready_out,

    input wire dram_rsp_valid_in, 
    input wire [AVS_DATAW-1:0] dram_rsp_data_in, 
    input wire [TAG_WIDTH-1:0] dram_rsp_tag_in,
    output wire dram_rsp_ready_in,    

    output wire dram_rsp_valid_out, 
    output wire [CCI_DATAW-1:0] dram_rsp_data_out, 
    output wire [TAG_WIDTH-1:0] dram_rsp_tag_out, 
    input wire dram_rsp_ready_out
);
    localparam N = AVS_ADDRW - CCI_ADDRW;

    `STATIC_ASSERT(N >= 0, ("oops!"))
    
    if (N == 0) begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign dram_req_valid_out  = dram_req_valid_in;
        assign dram_req_addr_out   = dram_req_addr_in;
        assign dram_req_rw_out     = dram_req_rw_in;
        assign dram_req_byteen_out = {AVS_BYTEENW{1'b1}};
        assign dram_req_data_out   = dram_req_data_in;
        assign dram_req_tag_out    = dram_req_tag_in; 
        assign dram_req_ready_in   = dram_req_ready_out;

        assign dram_rsp_valid_out  = dram_rsp_valid_in;
        assign dram_rsp_data_out   = dram_rsp_data_in;
        assign dram_rsp_tag_out    = dram_rsp_tag_in;
        assign dram_rsp_ready_in   = dram_rsp_ready_out;

    end else begin
        
        reg [N-1:0] req_ctr, rsp_ctr;

        wire [(2**N)-1:0][AVS_DATAW-1:0] dram_req_data_w_in;

        reg [(2**N)-1:0][AVS_DATAW-1:0] dram_rsp_data_r_out, dram_rsp_data_n_out;

        wire dram_req_fire_out = dram_req_valid_out && dram_req_ready_out;
        wire dram_rsp_fire_in = dram_rsp_valid_in && dram_rsp_ready_in;     

        assign dram_req_data_w_in = dram_req_data_in;

        always @(*) begin
            dram_rsp_data_n_out = dram_rsp_data_r_out;
            dram_rsp_data_n_out[rsp_ctr] = dram_rsp_data_in;
        end

        always @(posedge clk) begin
            if (reset) begin
                req_ctr <= 0;
                rsp_ctr <= 0;
            end else begin
                if (dram_req_fire_out) begin
                    req_ctr <= req_ctr + 1;
                end
                if (dram_rsp_fire_in) begin
                    rsp_ctr <= rsp_ctr + 1;
                    dram_rsp_data_r_out <= dram_rsp_data_n_out;
                end
            end 
        end

        assign dram_req_valid_out  = dram_req_valid_in;
        assign dram_req_addr_out   = {dram_req_addr_in, req_ctr};
        assign dram_req_rw_out     = dram_req_rw_in;
        assign dram_req_byteen_out = {AVS_BYTEENW{1'b1}};
        assign dram_req_data_out   = dram_req_data_w_in[req_ctr];
        assign dram_req_tag_out    = dram_req_tag_in; 
        assign dram_req_ready_in   = dram_req_ready_out && (req_ctr == (2**N-1));

        assign dram_rsp_valid_out  = dram_rsp_valid_in && (rsp_ctr == (2**N-1));
        assign dram_rsp_data_out   = dram_rsp_data_n_out;
        assign dram_rsp_tag_out    = dram_rsp_tag_in;
        assign dram_rsp_ready_in   = dram_rsp_ready_out;
    end

endmodule