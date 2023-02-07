/*
:set expandtab
:set tabstop=4
:set shiftwidth=4
:retab

*/


module ahb_mux
#(
    parameter ARBITRATION = "HIGH",
    parameter NMANAGERS=3
)
(
    input HCLK, HRESETn,
    ahb_if.subordinate m_in[NMANAGERS-1:0],
    ahb_if.manager m_out
);
    import ahb_pkg::*;
    import ahb_mux_pkg::*;
    
    logic [NMANAGERS-1:0] HREADY, HMASTLOCK;
    logic [NMANAGERS-1:0] [1:0] HTRANS;
    logic [NMANAGERS-1:0] ARB_SEL, ARB_SEL_PREV;
    logic [$clog2(NMANAGERS)-1:0] MASTER_SEL, MASTER_SEL_PREV;
    logic [NMANAGERS-1:0] [31:0] HWDATA_out;
    logic [NMANAGERS-1:0] [3:0] HWSTRB_out;
    aphase_t [NMANAGERS-1:0] downstreams;
    aphase_t downstream;

    arbiter #(.ARBITRATION(ARBITRATION), .NMANAGERS(NMANAGERS)) ARBITER(
        .HCLK,
        .HRESETn,
        .HTRANS,
        .HMASTLOCK,
        .HREADY,
        .ARB_SEL,
        .MASTER_SEL,
        .ARB_SEL_PREV,
        .MASTER_SEL_PREV
    );

    // Address Phase Master Mux
    assign downstream = downstreams[MASTER_SEL];

    // Output assignments
    assign m_out.HADDR = downstream.HADDR;
    assign m_out.HTRANS = downstream.HTRANS;
    assign m_out.HSIZE = downstream.HSIZE;
    assign m_out.HWRITE = downstream.HWRITE;
    assign m_out.HBURST = downstream.HBURST;
    assign m_out.HMASTLOCK = downstream.HMASTLOCK;

    assign m_out.HWDATA = HWDATA_out[MASTER_SEL_PREV];
    assign m_out.HWSTRB = HWSTRB_out[MASTER_SEL_PREV];

    genvar i;
    generate
        for(i = 0; i < NMANAGERS; i++) begin

            // Struct packing
            aphase_t upstream_in;
            assign upstream_in = {
                m_in[i].HADDR,
                m_in[i].HBURST,
                m_in[i].HMASTLOCK,
                m_in[i].HSIZE,
                m_in[i].HTRANS,
                m_in[i].HWRITE
            };

            // Array Packing
            assign HREADY[i] = m_in[i].HREADYOUT; // Does this make sense? Probably not
            assign HMASTLOCK[i] = downstreams[i].HMASTLOCK;
            assign HTRANS[i] = downstreams[i].HTRANS;

            // Cache instantiation
            aphase_cache CACHE(
                .HCLK,
                .HRESETn,
                .ARB_SEL(ARB_SEL[i]),
                .ARB_SEL_PREV(ARB_SEL_PREV[i]),
                .HREADY_in(m_out.HREADY),
                .HREADY_out(m_in[i].HREADYOUT),
                .upstream_in,
                .downstream_out(downstreams[i])
            );

            // Dphase signals
            assign m_in[i].HRESP = m_out.HRESP;
            assign m_in[i].HRDATA = m_out.HRDATA;
            assign HWDATA_out[i] = m_in[i].HWDATA;
            assign HWSTRB_out[i] = m_in[i].HWSTRB;
        end
    endgenerate


    `ifdef AHBL_BUS_MUX_FV
    // Reachability
    cov_ahbl_bus_mux_HTRANS_IDLE: cover property(@(posedge HCLK) HTRANS == IDLE);
    cov_ahbl_bus_mux_HTRANS_BUSY: cover property(@(posedge HCLK) HTRANS == BUSY);
    cov_ahbl_bus_mux_HTRANS_NONSEQ: cover property(@(posedge HCLK) HTRANS == NONSEQ);
    cov_ahbl_bus_mux_HTRANS_SEQ: cover property(@(posedge HCLK) HTRANS == SEQ);

    // Check reset value
    //property ARB_SEL_PREV_n_MASTER_SEL_PREV_RESET; @(posedge HCLK) disable iff(HRESETn)
    //    !HRESETn |=> (ARB_SEL_PREV == 1'b1 && MASTER_SEL_PREV == '0);
    //    endproperty
    //ast_ahbl_bus_mux_reset: assert property(ARB_SEL_PREV_n_MASTER_SEL_PREV_RESET);

    // Make sure that the master starts sending data when a grant signal is given
    //property BUS_GRANTED_AND_DATA_SENT(master); @(posedge HCLK)
    //    (HTRANS[master] == NONSEQ) |=> (HMASTLOCK=='0 && ARB_SEL[master] && MASTER_SEL==master);
    //    endproperty


    // Input Assumptions
    //asm_ahbl_bux_mux_input_HMASTLOCK: assume property(HMASTLOCK == 0);


    // When the master is not selected, signals are latched and HREADY = 0
    property MASTER_NOT_SELECTED(master); @(posedge HCLK)
        (HTRANS[master]==NONSEQ & !ARB_SEL[master] & !ARB_SEL_PREV[master]) |=> HREADY[master] == 0;
    endproperty


    // arbitor responses to master
    property ARBITOR_RESPONSE_SEL(master); @(posedge HCLK)
        HTRANS[master]==NONSEQ & (ARB_SEL[master]) ##[1:$] (ARB_SEL[master]);
    endproperty


    // slave responses to master control
    property SLAVE_RESPONSE_MASTER(master); @(posedge HCLK)
        HTRANS[master]==NONSEQ ##[1:$]
        ARB_SEL[master] ##[1:$]
        HREADY[master];
    endproperty



    // Assure it sends to the correct address

    property SIGNAL_CHECK_MASTER(master); @(posedge HCLK)
        (ARB_SEL[master]) |-> (m_out.HADDR == downstreams[master].HADDR
                              & m_out.HTRANS == downstreams[master].HTRANS
                              & m_out.HSIZE == downstreams[master].HSIZE
                              & m_out.HWRITE == downstreams[master].HWRITE
                              & m_out.HWSTRB == downstreams[master].HWSTRB
                              & m_out.HBURST == downstreams[master].HBURST
                              & m_out.HMASTLOCK == downstreams[master].HMASTLOCK);
    endproperty

    // Check HWDATA
    property DATA_CHECK_MASTER; @(posedge HCLK)
        //(HWDATA_out[MASTER_SEL_PREV] == m_out.HWDATA) & ((HWDATA_out[MASTER_SEL_PREV] != 0) & (HWDATA_out[MASTER_SEL_PREV] != 32'hffffffff)) |-> MASTER_SEL_PREV;
        MASTER_SEL_PREV |-> (HWDATA_out[MASTER_SEL_PREV] == m_out.HWDATA);
    endproperty
    ast_ahbl_bus_mux_data_check_master: assert property(DATA_CHECK_MASTER);

    // HREADY low wait state
    property HREADY_LOW_WAIT(master); @(posedge HCLK)
        (MASTER_SEL_PREV == master & !m_in[master].HREADY) |=> $stable(ARB_SEL_PREV);
    endproperty

    // Master's HREADY doesn't go high until the bus's HREADY goes high
    property MASTER_BUS_HREADY(master); @(posedge HCLK)
        (MASTER_SEL_PREV == master & m_out.HREADY) |-> m_in[master].HREADY;
    endproperty

    // Master Priority // These two properties only work if NMANAGERS==3
    //property MASTER_PRIORITY_02; @(posedge HCLK)
    //	(ARB_SEL_PREV[0]==1 & HMASTLOCK==0 & HTRANS[0]==NONSEQ & HTRANS[1]==NONSEQ & HTRANS[2]==NONSEQ) |-> (ARB_SEL[0]==0 & ARB_SEL[2]==1);
    //	endproperty
    //ast_ahbl_bus_mux_master_priority_02: assert property(MASTER_PRIORITY_02);

    //property MASTER_PRIORITY_12; @(posedge HCLK)
    //	(ARB_SEL_PREV[1]==1 & HMASTLOCK==0 & HTRANS[1]==NONSEQ & HTRANS[2]==NONSEQ) |-> (ARB_SEL[1]==0 & ARB_SEL[2]==1);
    //	endproperty
    //ast_ahbl_bus_mux_master_priority_12: assert property(MASTER_PRIORITY_12);




    // Data Phase: wait states depend on HREADY 
    property ADDR_PHASE_HREADY_CHECK(master); @(posedge HCLK)
        ((HTRANS[master]==NONSEQ | HTRANS[master]==SEQ) & ARB_SEL[master]) |=> HREADY[master] |-> ARB_SEL_PREV[master];
    endproperty


    generate
        for (genvar master = 0; master < NMANAGERS; master++) begin
            ast_ahbl_bus_mux_master_not_selected: assert property(MASTER_NOT_SELECTED(master));
            cov_ahbl_bus_mux_arbitor_response_to_master: cover property(ARBITOR_RESPONSE_SEL(master));
            cov_ahbl_bus_mux_slave_response_to_master: cover property(SLAVE_RESPONSE_MASTER(master));
            ast_ahbl_bus_mux_address_check_master: assert property(SIGNAL_CHECK_MASTER(master));
            ast_ahbl_bus_mux_hready_low_wait: assert property(HREADY_LOW_WAIT(master));
            ast_ahbl_bus_mux_master_bus_hready: assert property(MASTER_BUS_HREADY(master));
            ast_ahbl_bus_mux_addr_phase_hready_check: assert property(ADDR_PHASE_HREADY_CHECK(master));
        end
    endgenerate
    `endif

endmodule

