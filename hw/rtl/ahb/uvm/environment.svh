`ifndef ENVIROMENT_SVH
`define ENVIROMENT_SVH

import uvm_pkg::*;
`include "uvm_macros.svh"
`include "ahb_bus_agent.svh"
`include "ahb_if.sv"
`include "ahb_bus_scoreboard.svh"  // uvm_scoreboard
`include "ahb_bus_predictor.svh"  // uvm_subscriber
`include "ahb_bus_transaction_v2.svh"  // uvm_sequence_item

class environment extends uvm_env;
  `uvm_component_utils(environment)

  ahb_bus_agent ahb_agent;  // contains monitor and driver
  ahb_bus_predictor ahb_predictor;  // a reference model to check the result
  ahb_bus_scoreboard ahb_scoreboard;  // scoreboard

  function new(string name = "env", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    // instantiate all the components through factory method
    ahb_agent = ahb_bus_agent::type_id::create("ahb_agent", this);
    ahb_predictor = ahb_bus_predictor::type_id::create("ahb_predictor", this);
    ahb_scoreboard = ahb_bus_scoreboard::type_id::create("ahb_scoreboard", this);
  endfunction


  // TODO: Connect everything up correctly
  function void connect_phase(uvm_phase phase);
    ahb_agent.ahb_mon.ahb_bus_ap.connect(
        ahb_predictor.analysis_export);  // connect monitor to predictor
    ahb_predictor.pred_ap.connect(
        ahb_scoreboard.expected_export);  // connect predictor to comparator
    ahb_agent.ahb_mon.result_ap.connect(
        ahb_scoreboard.actual_export);  // connect monitor to comparator
  endfunction

endclass : environment

`endif
