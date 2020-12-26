`include "VX_platform.vh"

module VX_shift_register #( 
    parameter DATAW  = 1, 
    parameter RESETW = DATAW,
    parameter DEPTH  = 1
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);
    if (RESETW != 0) begin
        if (RESETW == DATAW) begin
    
            VX_shift_register_wr #(
                .DATAW (DATAW),
                .DEPTH (DEPTH)
            ) sr (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (data_in),
                .data_out (data_out)
            );
    
        end else begin
    
            VX_shift_register_wr #(
                .DATAW (DATAW),
                .DEPTH (DEPTH)
            ) sr_wr (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (data_in[DATAW-1:DATAW-RESETW]),
                .data_out (data_out[DATAW-1:DATAW-RESETW])
            );

            VX_shift_register_nr #(
                .DATAW (DATAW),
                .DEPTH (DEPTH)
            ) sr_nr (
                .clk      (clk),
                .enable   (enable),
                .data_in  (data_in[DATAW-RESETW-1:0]),
                .data_out (data_out[DATAW-RESETW-1:0])
            );

        end

    end else begin        

        `UNUSED_VAR (reset)
    
        VX_shift_register_nr #(
            .DATAW (DATAW),
            .DEPTH (DEPTH)
        ) sr (
            .clk      (clk),
            .enable   (enable),
            .data_in  (data_in),
            .data_out (data_out)
        );

    end    

endmodule

module VX_shift_register_nr #( 
    parameter DATAW  = 1,
    parameter DEPTH  = 1
) (
    input wire              clk,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);
    reg [DATAW-1:0] entries [DEPTH-1:0];

    always @(posedge clk) begin
        if (enable) begin                    
            for (integer i = DEPTH-1; i > 0; --i)
                entries[i] <= entries[i-1];
            entries[0] <= data_in;
        end
    end
    
    assign data_out = entries [DEPTH-1];

endmodule

module VX_shift_register_wr #( 
    parameter DATAW = 1, 
    parameter DEPTH = 1
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);
    reg [DEPTH-1:0][DATAW-1:0] entries;

    if (1 == DEPTH) begin

        always @(posedge clk) begin
            if (reset) begin
                entries <= (DEPTH * DATAW)'(0);
            end else begin
                if (enable) begin                    
                    entries <= data_in;
                end
            end
        end

    end else begin                
        
        always @(posedge clk) begin
            if (reset) begin
                entries <= (DEPTH * DATAW)'(0);
            end else begin
                if (enable) begin                    
                    entries <= {entries[DEPTH-2:0], data_in};
                end
            end
        end
    end    

    assign data_out = entries [DEPTH-1];

endmodule