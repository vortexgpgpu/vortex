`include "VX_platform.vh"

// Fast encoder using parallel prefix computation
// Adapter from BaseJump STL: http://bjump.org/data_out.html

module VX_onehot_encoder #(
    parameter N       = 1,    
    parameter REVERSE = 0,
    parameter FAST    = 1,
    parameter LN      = `LOG2UP(N)
) (
    input wire [N-1:0]   data_in,    
    output wire [LN-1:0] data_out,
    output wire          valid
); 
    if (N == 1) begin

        assign data_out = data_in;
        assign valid    = data_in;

    end else if (N == 2) begin

        assign data_out = data_in[!REVERSE];
        assign valid    = (| data_in);

    end else if (FAST) begin
    `IGNORE_WARNINGS_BEGIN
        localparam levels_lp = $clog2(N);
        localparam aligned_width_lp = 1 << $clog2(N);
    
        wire [levels_lp:0][aligned_width_lp-1:0] addr;
        wire [levels_lp:0][aligned_width_lp-1:0] v; 
    
        // base case, also handle padding for non-power of two inputs
        assign v[0] = REVERSE ? (data_in << (aligned_width_lp - N)) : ((aligned_width_lp)'(data_in));
        assign addr[0] = 'x;
    
        for (genvar level = 1; level < levels_lp+1; level=level+1) begin
            localparam segments_lp      = 2**(levels_lp-level);
            localparam segment_slot_lp  = aligned_width_lp/segments_lp;
            localparam segment_width_lp = level; // how many bits are needed at each level
        
            for (genvar segment = 0; segment < segments_lp; segment=segment+1) begin
                wire [1:0] vs = {
                    v[level-1][segment*segment_slot_lp+(segment_slot_lp >> 1)],
                    v[level-1][segment*segment_slot_lp]
                };
            
                assign v[level][segment*segment_slot_lp] = (| vs);

                if (level == 1) begin
                    assign addr[level][(segment*segment_slot_lp)+:segment_width_lp] = vs[!REVERSE]; 
                end else begin
                    assign addr[level][(segment*segment_slot_lp)+:segment_width_lp] = { 
                        vs[!REVERSE],
                        addr[level-1][segment*segment_slot_lp+:segment_width_lp-1] | addr[level-1][segment*segment_slot_lp+(segment_slot_lp >> 1)+:segment_width_lp-1]
                    };
                end        
            end  
        end	
    
        assign data_out = addr[levels_lp][`LOG2UP(N)-1:0];
        assign valid = (| data_in);
    `IGNORE_WARNINGS_END
    end else begin 

        reg [LN-1:0] index_r;

        if (REVERSE) begin

            always @(*) begin        
                index_r = 'x; 
                for (integer i = N-1; i >= 0; --i) begin
                    if (data_in[i]) begin                
                        index_r = `LOG2UP(N)'(i);
                    end
                end
            end

        end else begin
            always @(*) begin        
                index_r = 'x; 
                for (integer i = 0; i < N; i++) begin
                    if (data_in[i]) begin                
                        index_r = `LOG2UP(N)'(i);
                    end
                end
            end
        end

        assign data_out = index_r;
        assign valid    = (| data_in);
    end

endmodule