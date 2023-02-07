/*
*   Arbitration
*
*   HIGH - Highest-numbered master gets priority
*   LOW - Reverse-priority -- Lowest bit master gets priority
*/


module arbiter
#(
    parameter ARBITRATION = "HIGH",
    parameter NMANAGERS = 2
)
(
    input HCLK, HRESETn,
    input [NMANAGERS-1:0] [1:0] HTRANS,
    input [NMANAGERS-1:0] HMASTLOCK, HREADY,
    output logic [NMANAGERS-1:0] ARB_SEL, ARB_SEL_PREV,
    output logic [$clog2(NMANAGERS)-1:0] MASTER_SEL, MASTER_SEL_PREV
);
    import ahb_pkg::*;
    import ahb_mux_pkg::*;
    
    logic [NMANAGERS-1:0] ARB_SEL_PREV_n;
    logic [$clog2(NMANAGERS)-1:0] MASTER_SEL_PREV_n;

    always_ff @(posedge HCLK, negedge HRESETn) begin
        if(!HRESETn) begin
            ARB_SEL_PREV <= 'b1;
            MASTER_SEL_PREV <= '0;
        end else begin
            ARB_SEL_PREV <= ARB_SEL_PREV_n;
            MASTER_SEL_PREV <= MASTER_SEL_PREV_n;
        end
    end

    always_comb begin
        ARB_SEL_PREV_n = ARB_SEL_PREV;
        MASTER_SEL_PREV_n = MASTER_SEL_PREV;
        
        // This still works in the case of HMASTLOCK/HBURST,
        // since X_PREV_n will take the current ARB_SEL, which
        // in that case will have been set to the PREV value
        if(HREADY[MASTER_SEL_PREV]) begin
            ARB_SEL_PREV_n = ARB_SEL;
            MASTER_SEL_PREV_n = MASTER_SEL;
        end
    end

    always_comb begin
        if(HMASTLOCK[MASTER_SEL_PREV] == 'b1
            || HTRANS_t'(HTRANS[MASTER_SEL_PREV]) == BUSY
            || HTRANS_t'(HTRANS[MASTER_SEL_PREV]) == SEQ) begin

            ARB_SEL = ARB_SEL_PREV;
            MASTER_SEL = MASTER_SEL_PREV;
        end else begin
            //generate
            //    if(ARBITRATION == "HIGH") begin
                    //ARB_SEL = 'b1; // Default to low-prio master, always available
                    MASTER_SEL = '0;
                    for(int i = 0; i < NMANAGERS; i++) begin
                        if(HTRANS_t'(HTRANS[i]) != IDLE) begin
                            //ARB_SEL = '0;
                            //ARB_SEL[i] = 1'b1;
                            MASTER_SEL = i;
                        end
                    end
                    ARB_SEL = 'b1 << MASTER_SEL;
            /*    end else if(ARBITRATION == "LOW") begin
                    ARB_SEL[NMANAGERS-1] = 1'b1; // Default to low-prio master, always available
                    MASTER_SEL = NMANAGERS-1;;
                    for(int i = NMANAGERS-1; i >= 0; i--) begin
                        if(HTRANS[i] != IDLE) begin
                            ARB_SEL = '0;
                            ARB_SEL[i] = 1'b1;
                            MASTER_SEL = i;
                        end
                    end
                end
            endgenerate*/
        end
    end

endmodule
