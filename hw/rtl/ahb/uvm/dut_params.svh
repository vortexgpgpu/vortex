`ifndef DUT_PARAMS_SVH
`define DUT_PARAMS_SVH

`define AHB_ADDR_WIDTH 32
`define AHB_DATA_WIDTH 32

`define AHB_BASE_ADDR 0
`define AHB_NWORDS 512

`define AHB_NUM_ERROR_ADDR 5
`define AHB_NUM_WAIT_ADDR 5

parameter logic [`AHB_NUM_ERROR_ADDR-1:0][`AHB_ADDR_WIDTH-1:0] errorAddr = {
  32'h6, 32'h4, 32'h20, 32'h40, 32'h4c
};

parameter logic [`AHB_NUM_WAIT_ADDR-1:0][`AHB_ADDR_WIDTH-1:0] waitAddr = {
  32'h2, 32'h8, 32'h24, 32'h44, 32'h50
};

`endif
