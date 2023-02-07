

module aphase_cache ( 
    input HCLK,
    input HRESETn,
    input ARB_SEL,  // Select signal from arbiter
    input ARB_SEL_PREV,
    input HREADY_in, // HREADY from downstream bus
    input ahb_mux_pkg::aphase_t upstream_in,
    output logic HREADY_out, // HREADY going to master
    output ahb_mux_pkg::aphase_t downstream_out
);
    import ahb_mux_pkg::*;
    import ahb_pkg::*;
    
    logic valid, valid_n;
    aphase_t cache, cache_n;

    always_ff @(posedge HCLK, negedge HRESETn) begin
        if (!HRESETn) begin
            valid <= '0;
            cache <= '0;
        end else begin
            valid <= valid_n;
            cache <= cache_n;
        end
    end

    always_comb begin : APHASE
        // Kill cache once request is made,
        // Unless current Aphase is in the middle of
        // a waited transfer from request in Dphase
        if(ARB_SEL && HREADY_in) begin
            valid_n = 'b0;
            cache_n = '0;
        // Latch anytime we have a new request and don't have other data
        end else if(HTRANS_t'(upstream_in.HTRANS) != IDLE && !valid) begin
            valid_n = 'b1;
            cache_n = upstream_in;
        end else begin
            cache_n = cache;
            valid_n = valid;
        end
    end

    always_comb begin : SELECT
        if(!ARB_SEL_PREV && valid) begin
            HREADY_out = 1'b0;
        end else if(ARB_SEL_PREV) begin
            HREADY_out = HREADY_in;
        end else begin
            HREADY_out = 1'b1;
        end
    end

    assign downstream_out = (valid) ? cache : upstream_in;

endmodule

 
