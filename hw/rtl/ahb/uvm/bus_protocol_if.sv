`ifndef __BUS_PROTOCOL_IF__
`define __BUS_PROTOCOL_IF__

interface bus_protocol_if #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
) (  /* No I/O */
);

  // Vital signals
  logic wen;  // request is a data write
  logic ren;  // request is a data read
  logic request_stall;  // High when protocol should insert wait states in transaction
  logic [ADDR_WIDTH-1 : 0] addr; // *offset* address of request TODO: Is this good for general use?
  logic error;  // Indicate error condition to bus
  logic [(DATA_WIDTH/8)-1 : 0] strobe;  // byte enable for writes
  logic [DATA_WIDTH-1 : 0]
      wdata,
      rdata; // data lines -- from perspective of bus master. rdata should be data read from peripheral.

  // Hint signals
  logic is_burst;
  logic [1:0] burst_type;  // WRAP, INCR
  logic [7:0] burst_length;  // up to 256, would support AHB and AXI
  logic secure_transfer;  // TODO: How many bits?


  modport peripheral_vital(input wen, ren, addr, wdata, strobe, output rdata, error, request_stall);

  modport peripheral_hint(input is_burst, burst_type, burst_length, secure_transfer);

  modport protocol(
      input rdata, error, request_stall,
      output wen, ren, addr, wdata, strobe,  // vital signals
      is_burst, burst_type, burst_length, secure_transfer  // hint signals
  );

endinterface
`endif
