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

    input wire mem_req_valid_in,
    input wire [CCI_ADDRW-1:0] mem_req_addr_in,
    input wire mem_req_rw_in,
    input wire [CCI_DATAW-1:0] mem_req_data_in,
    input wire [TAG_WIDTH-1:0] mem_req_tag_in,
    output wire mem_req_ready_in,

    output wire mem_req_valid_out,
    output wire [AVS_ADDRW-1:0] mem_req_addr_out,
    output wire mem_req_rw_out,
    output wire [AVS_BYTEENW-1:0] mem_req_byteen_out,
    output wire [AVS_DATAW-1:0] mem_req_data_out,
    output wire [TAG_WIDTH-1:0] mem_req_tag_out,
    input wire mem_req_ready_out,

    input wire mem_rsp_valid_in, 
    input wire [AVS_DATAW-1:0] mem_rsp_data_in, 
    input wire [TAG_WIDTH-1:0] mem_rsp_tag_in,
    output wire mem_rsp_ready_in,    

    output wire mem_rsp_valid_out, 
    output wire [CCI_DATAW-1:0] mem_rsp_data_out, 
    output wire [TAG_WIDTH-1:0] mem_rsp_tag_out, 
    input wire mem_rsp_ready_out
);
    localparam N = AVS_ADDRW - CCI_ADDRW;

    `STATIC_ASSERT(N >= 0, ("oops!"))
    
    if (N == 0) begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign mem_req_valid_out  = mem_req_valid_in;
        assign mem_req_addr_out   = mem_req_addr_in;
        assign mem_req_rw_out     = mem_req_rw_in;
        assign mem_req_byteen_out = {AVS_BYTEENW{1'b1}};
        assign mem_req_data_out   = mem_req_data_in;
        assign mem_req_tag_out    = mem_req_tag_in; 
        assign mem_req_ready_in   = mem_req_ready_out;

        assign mem_rsp_valid_out  = mem_rsp_valid_in;
        assign mem_rsp_data_out   = mem_rsp_data_in;
        assign mem_rsp_tag_out    = mem_rsp_tag_in;
        assign mem_rsp_ready_in   = mem_rsp_ready_out;

    end else begin
        
        reg [N-1:0] req_ctr, rsp_ctr;

        wire [(2**N)-1:0][AVS_DATAW-1:0] mem_req_data_w_in;

        reg [(2**N)-1:0][AVS_DATAW-1:0] mem_rsp_data_r_out, mem_rsp_data_n_out;

        wire mem_req_fire_out = mem_req_valid_out && mem_req_ready_out;
        wire mem_rsp_fire_in = mem_rsp_valid_in && mem_rsp_ready_in;     

        assign mem_req_data_w_in = mem_req_data_in;

        always @(*) begin
            mem_rsp_data_n_out = mem_rsp_data_r_out;
            mem_rsp_data_n_out[rsp_ctr] = mem_rsp_data_in;
        end

        always @(posedge clk) begin
            if (reset) begin
                req_ctr <= 0;
                rsp_ctr <= 0;
            end else begin
                if (mem_req_fire_out) begin
                    req_ctr <= req_ctr + 1;
                end
                if (mem_rsp_fire_in) begin
                    rsp_ctr <= rsp_ctr + 1;
                    mem_rsp_data_r_out <= mem_rsp_data_n_out;
                end
            end 
        end

        assign mem_req_valid_out  = mem_req_valid_in;
        assign mem_req_addr_out   = {mem_req_addr_in, req_ctr};
        assign mem_req_rw_out     = mem_req_rw_in;
        assign mem_req_byteen_out = {AVS_BYTEENW{1'b1}};
        assign mem_req_data_out   = mem_req_data_w_in[req_ctr];
        assign mem_req_tag_out    = mem_req_tag_in; 
        assign mem_req_ready_in   = mem_req_ready_out && (req_ctr == (2**N-1));

        assign mem_rsp_valid_out  = mem_rsp_valid_in && (rsp_ctr == (2**N-1));
        assign mem_rsp_data_out   = mem_rsp_data_n_out;
        assign mem_rsp_tag_out    = mem_rsp_tag_in;
        assign mem_rsp_ready_in   = mem_rsp_ready_out;
    end

endmodule