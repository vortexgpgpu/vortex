
module ahb_simple_interconnect #(
    parameter int NSUBORDINATES = 2,
    parameter logic [31:0] AHB_MAP[NSUBORDINATES] = '{default: '0}
) (
    // modports reversed -- interconnect looks like subordinate
    // from perspective of e.g. RISC-V core, but looks like manager
    // form perspective of e.g. memory 
    ahb_if.subordinate manager,
    ahb_if.manager subordinates[NSUBORDINATES]
);

    logic [31:0] HRDATA_arr [NSUBORDINATES];
    logic HRESP_arr [NSUBORDINATES];
    logic HREADY_arr [NSUBORDINATES];
    logic addr_match_arr [NSUBORDINATES];

    logic [$clog2(NSUBORDINATES):0] dphase_select, dphase_select_n;

    genvar i;

    function logic addr_match(logic [31:0] bus_addr, int subordinate_idx);
        return (bus_addr >= AHB_MAP[subordinate_idx] && (subordinate_idx == NSUBORDINATES-1 || bus_addr < AHB_MAP[subordinate_idx + 1]));
    endfunction

    generate
        for(i = 0; i < NSUBORDINATES; i++) begin: g_addr_match
            assign addr_match_arr[i] = addr_match(manager.HADDR, i);
        end
    endgenerate

    generate
        for(i = 0; i < NSUBORDINATES; i++) begin : g_request
            assign subordinates[i].HWRITE       = manager.HWRITE;
            assign subordinates[i].HMASTLOCK    = manager.HMASTLOCK;
            assign subordinates[i].HTRANS       = manager.HTRANS;
            assign subordinates[i].HBURST       = manager.HBURST;
            assign subordinates[i].HSIZE        = manager.HSIZE;
            assign subordinates[i].HADDR        = manager.HADDR;
            assign subordinates[i].HWDATA       = manager.HWDATA;
            assign subordinates[i].HWSTRB       = manager.HWSTRB;
            assign subordinates[i].HSEL         = (manager.HTRANS != ahb_pkg::IDLE && addr_match_arr[i]);
        end
    endgenerate

    always_ff @(posedge manager.HCLK, negedge manager.HRESETn) begin
        if(!manager.HRESETn) begin
            dphase_select <= '0;
        end else begin
            dphase_select <= dphase_select_n;
        end
    end

    // dphase select: if HREADY low, stall. Otherwise, always
    // safe to take the next matching address since it will be
    // ignored anyways if there isn't a useful transaction
    always_comb begin
        //manager.HRDATA = 32'hBAD1BAD1;
        //manager.HREADYOUT = 1'b1;
        //manager.HRESP  = 1'b1;
        dphase_select_n = dphase_select;

        for(int i = 0; i < NSUBORDINATES; i++) begin
            //if(addr_match_arr[i]) begin
            //    manager.HREADYOUT = HREADY_arr[dphase_select];
            //    manager.HRDATA = HRDATA_arr[dphase_select];
            //    manager.HRESP  = HRESP_arr[dphase_select];
            //    if(manager.HREADYOUT) begin
            //        dphase_select_n = i; 
            //    end
            //end
            if(manager.HREADYOUT && addr_match_arr[i]) begin
                dphase_select_n = i;
            end
        end
    end

    // NOTE: This depends on subordinates correctly asserting HREADYOUT
    // while idle. 
    assign manager.HREADYOUT = HREADY_arr[dphase_select];
    assign manager.HRDATA    = HRDATA_arr[dphase_select];
    assign manager.HRESP     = HRESP_arr[dphase_select];

    generate
        for(i = 0; i < NSUBORDINATES; i++) begin
            assign HREADY_arr[i] = subordinates[i].HREADY;
            assign HRDATA_arr[i] = subordinates[i].HRDATA;
            assign HRESP_arr[i]  = subordinates[i].HRESP;
        end
    endgenerate

endmodule
