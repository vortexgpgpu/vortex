
module ahb_mux_wrapper #(
    parameter int NSUBORDINATES = 3    
)(
    input HCLK,
    input HRESETn,
    input HSEL_in[NSUBORDINATES], 
    input HWRITE_in[NSUBORDINATES],
    input HMASTLOCK_in[NSUBORDINATES],
    input [1:0] HTRANS_in[NSUBORDINATES],
    input [2:0] HBURST_in[NSUBORDINATES],
    input [2:0] HSIZE_in[NSUBORDINATES],
    input [31:0] HADDR_in[NSUBORDINATES],
    input [31:0] HWDATA_in[NSUBORDINATES],
    input [3:0] HWSTRB_in[NSUBORDINATES],

    output logic HSEL_out, 
    output logic HWRITE_out,
    output logic HMASTLOCK_out,
    output logic [1:0] HTRANS_out,
    output logic [2:0] HBURST_out,
    output logic [2:0] HSIZE_out,
    output logic [31:0] HADDR_out,
    output logic [31:0] HWDATA_out,
    output logic [3:0] HWSTRB_out,

    output logic HREADYOUT_in[NSUBORDINATES],
    output logic [31:0] HRDATA_in[NSUBORDINATES],
    output logic HRESP_in[NSUBORDINATES],

    input HREADY_out,
    input [31:0] HRDATA_out,
    input HRESP_out
);

    ahb_if managers[NSUBORDINATES](HCLK, HRESETn);
    ahb_if muxed_if(HCLK, HRESETn);


    genvar i;

    generate
        for(i = 0; i < NSUBORDINATES; i++) begin
            assign managers[i].HSEL = HSEL_in[i];
            assign managers[i].HWRITE = HWRITE_in[i];
            assign managers[i].HMASTLOCK = HMASTLOCK_in[i];
            assign managers[i].HTRANS = HTRANS_in[i];
            assign managers[i].HBURST = HBURST_in[i];
            assign managers[i].HSIZE = HSIZE_in[i];
            assign managers[i].HADDR = HADDR_in[i];
            assign managers[i].HWDATA = HWDATA_in[i];
            assign managers[i].HWSTRB = HWSTRB_in[i];
            assign HREADYOUT_in[i] = managers[i].HREADYOUT;
            assign HRDATA_in[i] = managers[i].HRDATA;
            assign HRESP_in[i] = managers[i].HRESP;
        end
    endgenerate


    assign HSEL_out      =muxed_if.HSEL       ;
    assign HWRITE_out    =muxed_if.HWRITE     ;
    assign HMASTLOCK_out =muxed_if.HMASTLOCK  ;
    assign HTRANS_out    =muxed_if.HTRANS     ;
    assign HBURST_out    =muxed_if.HBURST     ;
    assign HSIZE_out     =muxed_if.HSIZE      ;
    assign HADDR_out     =muxed_if.HADDR      ;
    assign HWDATA_out    =muxed_if.HWDATA     ;
    assign HWSTRB_out    =muxed_if.HWSTRB     ;

    assign muxed_if.HREADYOUT = HREADYOUT_in;
    assign muxed_if.HRESP = HRESP_in;
    assign muxed_if.HRDATA = HRDATA_in;


endmodule

