`include "VX_platform.vh"

`TRACING_OFF
module VX_skid_buffer #(
    parameter DATAW          = 1,
    parameter PASSTHRU       = 0,
    parameter NOBACKPRESSURE = 0,
    parameter OUT_REG        = 0
) ( 
    input  wire             clk,
    input  wire             reset,
    
    input  wire             valid_in,
    output wire             ready_in,        
    input  wire [DATAW-1:0] data_in,

    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out
);

    if (PASSTHRU) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end else if (NOBACKPRESSURE) begin

        `RUNTIME_ASSERT(ready_out, ("%t: *** ready_out should always be asserted", $time))

        wire stall = valid_out && ~ready_out;

        VX_pipe_register #(
            .DATAW  (1 + DATAW),
            .RESETW (1)
        ) pipe_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (!stall),
            .data_in  ({valid_in, data_in}),
            .data_out ({valid_out, data_out})
        );

        assign ready_in = ~stall;
    
    end else begin

        if (OUT_REG) begin

            reg [DATAW-1:0] data_out_r;
            reg [DATAW-1:0] buffer;
            reg             valid_out_r;
            reg             use_buffer;
            
            wire push = valid_in && ready_in;
            wire pop = !valid_out_r || ready_out;
            
            always @(posedge clk) begin
                if (reset) begin
                    valid_out_r <= 0; 
                    use_buffer  <= 0;
                end else begin             
                    if (ready_out) begin
                        use_buffer <= 0;
                    end else if (valid_in && valid_out_r) begin
                        use_buffer <= 1;
                    end
                    if (pop) begin
                        valid_out_r <= valid_in || use_buffer;
                    end
                end
            end

            always @(posedge clk) begin
                if (push) begin
                    buffer <= data_in;
                end
                if (pop && !use_buffer) begin
                    data_out_r <= data_in;                    
                end else if (ready_out) begin
                    data_out_r <= buffer;
                end
            end

            assign ready_in  = !use_buffer;
            assign valid_out = valid_out_r;
            assign data_out  = data_out_r;

        end else begin

            reg [DATAW-1:0] shift_reg [1:0];
            reg valid_out_r, ready_in_r, rd_ptr_r;

            wire push = valid_in && ready_in;
            wire pop = valid_out_r && ready_out;

            always @(posedge clk) begin
                if (reset) begin
                    valid_out_r <= 0;
                    ready_in_r  <= 1;
                    rd_ptr_r    <= 1;
                end else begin
                    if (push) begin
                        if (!pop) begin                            
                            ready_in_r  <= rd_ptr_r;
                            valid_out_r <= 1;
                        end
                    end else if (pop) begin
                        ready_in_r  <= 1;
                        valid_out_r <= rd_ptr_r;
                    end
                    rd_ptr_r <= rd_ptr_r ^ (push ^ pop);
                end                   
            end

            always @(posedge clk) begin
                if (push) begin
                    shift_reg[1] <= shift_reg[0];
                    shift_reg[0] <= data_in;
                end
            end

            assign ready_in  = ready_in_r;
            assign valid_out = valid_out_r;
            assign data_out  = shift_reg[rd_ptr_r];
        end
    end

endmodule
`TRACING_ON
