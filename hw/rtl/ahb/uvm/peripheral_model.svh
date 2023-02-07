`ifndef PERIPHERAL_MODEL_SVH
`define PERIPHERAL_MODEL_SVH

`include "bus_protocol_if.sv"
`include "dut_params.svh"

class peripheral_model;

  logic [`AHB_NWORDS-1:0][`AHB_DATA_WIDTH-1:0] words;


  function void writeWord(logic [`AHB_DATA_WIDTH-1:0] data,
                          logic [(`AHB_DATA_WIDTH / 8)-1:0] strb,
                          logic [`AHB_ADDR_WIDTH-1:0] addr);

    logic [`AHB_DATA_WIDTH-1:0] expandedStrobe;
    expandedStrobe = '0;
    /*for (int i = 0; i < `AHB_DATA_WIDTH / 8; i++) begin
      expandedStrobe[7:0] |= {8{strb[i]}};
      if (i != (`AHB_DATA_WIDTH / 8) - 1) begin
        expandedStrobe = expandedStrobe << 8;
      end
    end*/
    for (int i = (`AHB_DATA_WIDTH / 8) - 1; i >= 0; i--) begin
      expandedStrobe[7:0] |= {8{strb[i]}};
      if (i != 0) begin
        expandedStrobe = expandedStrobe << 8;
      end
    end
    //`uvm_info("model", $sformatf("Model expanded strove:\n%h", expandedStrobe), UVM_LOW);
    words[addr[$clog2(`AHB_NWORDS)-1+$clog2(`AHB_DATA_WIDTH/8):$clog2(`AHB_DATA_WIDTH/8)]] = data &
        expandedStrobe;

  endfunction

  function logic [`AHB_DATA_WIDTH-1:0] readWord(logic [`AHB_ADDR_WIDTH-1:0] addr);
    return words[addr[$clog2(`AHB_NWORDS)-1+$clog2(`AHB_DATA_WIDTH/8):$clog2(`AHB_DATA_WIDTH/8)]];

  endfunction



endclass

`endif
