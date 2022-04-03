`include "VX_define.vh"

`TRACING_OFF
module VX_mem_rsp_sel #(
    parameter NUM_REQS      = 1, 
    parameter DATA_WIDTH    = 1, 
    parameter TAG_WIDTH     = 1,    
    parameter TAG_SEL_BITS  = 0,
    parameter OUT_REG       = 0
) (
    input wire              clk,
    input wire              reset,

    // input response
    input wire [NUM_REQS-1:0]                   rsp_valid_in,
    input wire [NUM_REQS-1:0][DATA_WIDTH-1:0]   rsp_data_in,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0]    rsp_tag_in,
    output wire [NUM_REQS-1:0]                  rsp_ready_in,

    // output responses
    output wire                             rsp_valid_out,
    output wire [NUM_REQS-1:0]              rsp_tmask_out,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0] rsp_data_out,
    output wire [TAG_WIDTH-1:0]             rsp_tag_out,
    input wire                              rsp_ready_out
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
            .requests     (rsp_valid_in), 
            .grant_valid  (grant_valid),
            .grant_index  (grant_index),
            `UNUSED_PIN (grant_onehot)            
        );

        reg [NUM_REQS-1:0] rsp_valid_sel;
        reg [NUM_REQS-1:0] rsp_ready_sel;
        wire rsp_ready_unqual;

        wire [TAG_WIDTH-1:0] rsp_tag_sel = rsp_tag_in[grant_index];
        
        always @(*) begin                
            rsp_valid_sel = 0;              
            rsp_ready_sel = 0;
            
            for (integer i = 0; i < NUM_REQS; i++) begin
                if (rsp_tag_in[i][TAG_SEL_BITS-1:0] == rsp_tag_sel[TAG_SEL_BITS-1:0]) begin
                    rsp_valid_sel[i] = rsp_valid_in[i];                    
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
            .data_in   ({rsp_valid_sel, rsp_tag_sel, rsp_data_in}),
            .ready_in  (rsp_ready_unqual),      
            .valid_out (rsp_valid_out),
            .data_out  ({rsp_tmask_out, rsp_tag_out, rsp_data_out}),
            .ready_out (rsp_ready_out)
        );  

        assign rsp_ready_in = rsp_ready_sel;     
        
    end else begin

        assign rsp_valid_out = rsp_valid_in;
        assign rsp_tmask_out = 1'b1;
        assign rsp_tag_out   = rsp_tag_in;
        assign rsp_data_out  = rsp_data_in;
        assign rsp_ready_in = rsp_ready_out;

    end

endmodule
`TRACING_ON