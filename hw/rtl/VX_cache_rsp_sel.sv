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

    localparam LOG_NUM_REQS = `CLOG2(NUM_REQS);

    if (NUM_REQS > 1) begin

        wire [LOG_NUM_REQS-1:0] grant_index;
        wire grant_valid;
        wire rsp_fire;

        VX_rr_arbiter #(
            .NUM_REQS    (NUM_REQS),
            .LOCK_ENABLE (1)
        ) arbiter (
            .clk          (clk),
            .reset        (reset),
            .unlock       (rsp_fire),
            .requests     (rsp_in_if.valid), 
            .grant_valid  (grant_valid),
            .grant_index  (grant_index),
            `UNUSED_PIN (grant_onehot)            
        );

        reg [NUM_REQS-1:0] rsp_valid_sel;
        reg [NUM_REQS-1:0] rsp_ready_sel;
        wire rsp_ready_unqual;

        wire [TAG_WIDTH-1:0] rsp_tag_sel = rsp_in_if.tag[grant_index];
        
        always @(*) begin                
            rsp_valid_sel = 0;              
            rsp_ready_sel = 0;
            
            for (integer i = 0; i < NUM_REQS; i++) begin
                if (rsp_in_if.tag[i][TAG_SEL_BITS-1:0] == rsp_tag_sel[TAG_SEL_BITS-1:0]) begin
                    rsp_valid_sel[i] = rsp_in_if.valid[i];                    
                    rsp_ready_sel[i] = rsp_ready_unqual;
                end
            end
        end                            

        assign rsp_fire = grant_valid && rsp_ready_unqual;
        
        VX_skid_buffer #(
            .DATAW    (NUM_REQS + TAG_WIDTH + (NUM_REQS * DATA_WIDTH)),
            .PASSTHRU (0 == OUT_REG)
        ) out_sbuf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (grant_valid),        
            .data_in   ({rsp_valid_sel, rsp_tag_sel, rsp_in_if.data}),
            .ready_in  (rsp_ready_unqual),      
            .valid_out (rsp_out_valid),
            .data_out  ({rsp_out_tmask, rsp_out_tag, rsp_out_data}),
            .ready_out (rsp_out_ready)
        );  

        assign rsp_in_if.ready = rsp_ready_sel;     
        
    end else begin

        assign rsp_out_valid = rsp_in_if.valid;
        assign rsp_out_tmask = 1'b1;
        assign rsp_out_tag   = rsp_in_if.tag;
        assign rsp_out_data  = rsp_in_if.data;
        assign rsp_in_if.ready = rsp_out_ready;

    end

endmodule