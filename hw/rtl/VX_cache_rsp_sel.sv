`include "VX_define.vh"

module VX_cache_rsp_sel #(
    parameter NUM_REQS      = 1, 
    parameter DATA_WIDTH    = 1, 
    parameter TAG_WIDTH     = 1,    
    parameter TAG_SEL_BITS  = 0,
    parameter OUT_REG       = 0
) (
    input wire              clk,
    input wire              reset,

    // input response
    VX_cache_rsp_if.slave   rsp_in_if,

    // output responses
    output wire                             rsp_out_valid,
    output wire [NUM_REQS-1:0]              rsp_out_tmask,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0] rsp_out_data,
    output wire [TAG_WIDTH-1:0]             rsp_out_tag,
    input wire                              rsp_out_ready
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    if (NUM_REQS > 1) begin

        reg [NUM_REQS-1:0] rsp_valid_unqual;
        wire [TAG_WIDTH-1:0] rsp_tag_unqual;
        wire rsp_ready_unqual;

        reg [NUM_REQS-1:0] rsp_in_ready_r;

        VX_find_first #(
            .N     (NUM_REQS),
            .DATAW (TAG_WIDTH)
        ) find_first (
            .valid_i (rsp_in_if.valid),
            .data_i  (rsp_in_if.tag),
            .data_o  (rsp_tag_unqual),
            `UNUSED_PIN (valid_o)
        );
        
        always @(*) begin                
            rsp_valid_unqual = 0;              
            rsp_in_ready_r   = 0;
            
            for (integer i = 0; i < NUM_REQS; i++) begin
                if (rsp_in_if.tag[i][TAG_SEL_BITS-1:0] == rsp_tag_unqual[TAG_SEL_BITS-1:0]) begin
                    rsp_valid_unqual[i] = rsp_in_if.valid[i];                    
                    rsp_in_ready_r[i] = rsp_ready_unqual;
                end
            end
        end                            

        wire rsp_valid_any = (| rsp_in_if.valid);
        
        VX_skid_buffer #(
            .DATAW    (NUM_REQS + TAG_WIDTH + (NUM_REQS * DATA_WIDTH)),
            .PASSTHRU (0 == OUT_REG)
        ) out_sbuf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rsp_valid_any),        
            .data_in   ({rsp_valid_unqual, rsp_tag_unqual, rsp_in_if.data}),
            .ready_in  (rsp_ready_unqual),      
            .valid_out (rsp_out_valid),
            .data_out  ({rsp_out_tmask, rsp_out_tag, rsp_out_data}),
            .ready_out (rsp_out_ready)
        );  

        assign rsp_in_if.ready = rsp_in_ready_r;     
        
    end else begin

        assign rsp_out_valid = rsp_in_if.valid;
        assign rsp_out_tmask = 1'b1;
        assign rsp_out_tag   = rsp_in_if.tag;
        assign rsp_out_data  = rsp_in_if.data;
        assign rsp_in_if.ready = rsp_out_ready;

    end

endmodule
