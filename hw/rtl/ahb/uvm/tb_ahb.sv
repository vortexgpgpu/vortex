`include "ahb_subordinate.sv"
`include "peripheral_bfm.sv"

// interface file
`include "ahb_if.sv"
`include "bus_protocol_if.sv"

// UVM test file
`include "testAll.svh"

// Include params
`include "dut_params.svh"

`timescale 1ns / 1ps
// import uvm packages
import uvm_pkg::*;

module tb_ahb ();
  logic clk;

  // generate clock
  initial begin
    clk = 0;
    forever #10 clk = !clk;
  end
  assign ahbif.HCLK = clk;

  // instantiate the interface
  ahb_if #(
      .ADDR_WIDTH(`AHB_ADDR_WIDTH),
      .DATA_WIDTH(`AHB_DATA_WIDTH)
  ) ahbif ();

  bus_protocol_if #(
      .ADDR_WIDTH(`AHB_ADDR_WIDTH),
      .DATA_WIDTH(`AHB_DATA_WIDTH)
  ) bpif ();

  peripheral_bfm periphBFM (
      ahbif.HCLK,
      ahbif.HRESETn,
      bpif
  );

  // TODO: instantiate the DUT
  ahb_subordinate #(
      .BASE_ADDR(`AHB_BASE_ADDR),
      .NWORDS(`AHB_NWORDS)
  ) ahb_mod (
      ahbif,
      bpif
  );

  initial begin
    uvm_config_db#(virtual ahb_if)::set(null, "", "ahb_vif",
                                        ahbif); // configure the interface into the database, so that it can be accessed throughout the hierachy
    run_test("testAll");
  end
endmodule
