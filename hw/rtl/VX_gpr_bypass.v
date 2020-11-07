`include "VX_platform.vh"

module VX_gpr_bypass #(
    parameter DATAW    = 1,
    parameter PASSTHRU = 0
) ( 
    input  wire             clk,
    input  wire             reset,
    input  wire             push,    
    input  wire             pop,
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out
); 
    if (PASSTHRU) begin
        reg delayed_push;
        
        always @(posedge clk) begin
            if (reset) begin
                delayed_push <= 0;
            end else begin
                delayed_push <= push;    
                assert(!delayed_push || pop);
            end
        end

        assign data_out = data_in;

    end else begin

        reg [DATAW-1:0] buffer, buffer2;
        reg use_buffer, use_buffer2;
        reg delayed_push;

        always @(posedge clk) begin
            if (reset) begin
                delayed_push <= 0;
                use_buffer   <= 0;
                use_buffer2  <= 0;
                buffer       <= 0;            
                buffer2      <= 0;
            end else begin  
                delayed_push <= push;    
                assert(!use_buffer2 || use_buffer); 
                if (pop) begin
                    buffer      <= buffer2;
                    use_buffer  <= use_buffer2;
                    use_buffer2 <= 0;
                end          
                if (delayed_push) begin                
                    if (use_buffer) begin
                        assert(!use_buffer2); // full!
                        use_buffer <= 1;
                        if (pop) begin
                            buffer <= data_in;
                        end else begin                        
                            buffer2 <= data_in;
                            use_buffer2 <= 1; 
                        end
                    end else if (!pop) begin
                        buffer <= data_in;
                        use_buffer <= 1;                        
                    end
                end
            end
        end

        assign data_out = use_buffer ? buffer : data_in;
    end

endmodule