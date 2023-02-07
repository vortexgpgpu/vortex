`include "bus_protocol_if.sv"
`include "dut_params.svh"

module peripheral_bfm (
    input logic CLK,
    input logic nRST,
    bus_protocol_if bpif
);

  localparam dataIndexWidth = $clog2(`AHB_NWORDS);

  logic [`AHB_NWORDS-1:0][`AHB_DATA_WIDTH-1:0] data;
  logic [`AHB_NWORDS-1:0][`AHB_DATA_WIDTH-1:0] ndata;
  logic [`AHB_NWORDS-1:0][`AHB_DATA_WIDTH-1:0] nndata;  // used for wait states
  logic [`AHB_NWORDS-1:0][`AHB_DATA_WIDTH-1:0] waitData;

  logic [`AHB_DATA_WIDTH-1:0] expandedStrobe;
  logic [`AHB_DATA_WIDTH-1:0] nrdata;
  logic [`AHB_DATA_WIDTH-1:0] rdata;

  logic waited;
  logic nwaited;
  logic errorOccured;

  always_ff @(posedge CLK, negedge nRST) begin
    if (~nRST) begin
      data <= '0;
      waited <= 0;
      waitData <= '0;
      rdata <= '0;
    end else begin
      rdata <= nrdata;
      waitData <= nndata;
      data <= ndata;
      waited <= nwaited;
    end
  end


  always_comb begin
    ndata = data;
    bpif.rdata = '0;
    bpif.error = 0;
    bpif.request_stall = 0;
    nwaited = 0;
    errorOccured = 0;
    expandedStrobe = '0;
    nrdata = '0;
    nndata = data;



    for (int i = 0; i < `AHB_NUM_ERROR_ADDR; i++) begin
      if (errorAddr[i] == bpif.addr) begin
        bpif.error   = 1;
        errorOccured = 1;
      end
    end

    for (int i = 0; i < `AHB_NUM_WAIT_ADDR; i++) begin
      if (waitAddr[i] == bpif.addr && !waited) begin
        nwaited = 1;
        bpif.request_stall = 1;
      end
    end

    if (bpif.ren && !nwaited && !errorOccured) begin
      bpif.rdata =
          data[bpif.addr[dataIndexWidth-1+$clog2(`AHB_DATA_WIDTH/8):$clog2(`AHB_DATA_WIDTH/8)]];
    end else if (bpif.wen && !nwaited && !errorOccured) begin
      for (int i = (`AHB_DATA_WIDTH / 8) - 1; i >= 0; i--) begin
        expandedStrobe[7:0] |= {8{bpif.strobe[i]}};
        if (i != 0) begin
          expandedStrobe = expandedStrobe << 8;
        end
      end
      ndata[bpif.addr[dataIndexWidth-1+$clog2(`AHB_DATA_WIDTH/8):$clog2(`AHB_DATA_WIDTH/8)]] =
          bpif.wdata & expandedStrobe;
    end else if (bpif.ren && nwaited) begin
      nrdata =
          data[bpif.addr[dataIndexWidth-1+$clog2(`AHB_DATA_WIDTH/8):$clog2(`AHB_DATA_WIDTH/8)]];
    end else if (bpif.wen && nwaited) begin
      for (int i = (`AHB_DATA_WIDTH / 8) - 1; i >= 0; i--) begin
        expandedStrobe[7:0] |= {8{bpif.strobe[i]}};
        if (i != 0) begin
          expandedStrobe = expandedStrobe << 8;
        end
      end
      nndata[bpif.addr[dataIndexWidth-1+$clog2(`AHB_DATA_WIDTH/8):$clog2(`AHB_DATA_WIDTH/8)]] =
          bpif.wdata & expandedStrobe;
    end

    if (waited) begin
      ndata = waitData;
      bpif.rdata = rdata;
      bpif.error = 0;
      bpif.request_stall = 0;
    end
  end



endmodule
