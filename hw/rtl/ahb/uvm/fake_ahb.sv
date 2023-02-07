`include "ahb_if.sv"

module fake_ahb (
    input logic CLK,
    ahb_if.subordinate ahbif
);

  logic [31:0][127:0] dataArray;
  logic [31:0][127:0] ndataArray;

  logic [31:0] nrdata;
  logic nhready;
  logic nhresp;
  logic [31:0] nwdata;
  logic writeFlag, nwriteFlag;

  always_ff @(posedge CLK, negedge ahbif.HRESETn) begin
    if (~ahbif.HRESETn) begin
      dataArray       <= {128{32'h12345678}};
      ahbif.HREADYOUT <= '0;
      ahbif.HRDATA    <= '0;
      ahbif.HRESP     <= '0;
      writeFlag       <= 0;
      nwdata          <= '0;
    end else begin
      dataArray       <= ndataArray;
      ahbif.HREADYOUT <= nhready;
      ahbif.HRDATA    <= nrdata;
      ahbif.HRESP     <= nhresp;
      writeFlag       <= nwriteFlag;
      nwdata          <= ahbif.HWDATA;
    end
  end

  always_comb begin
    nwriteFlag = ahbif.HWRITE && ahbif.HSEL && ahbif.HTRANS == 2; // only write if wrtie, selected and not idle transfer
    ndataArray = dataArray;
    nhready = ahbif.HSEL;  // provides zero wait state okay response to IDLE as well
    nhresp = '0;  //Never have an error condition
    nrdata = '0;

    if (writeFlag) begin
      ndataArray[ahbif.HADDR[8:2]] = nwdata;
    end

    if(!ahbif.HWRITE && ahbif.HSEL && ahbif.HTRANS == 2) begin // only read if selected, not writing, and a nonseq (single) trasnfer for basic tests
      nrdata = dataArray[ahbif.HADDR[8:2]];
    end
  end

endmodule
