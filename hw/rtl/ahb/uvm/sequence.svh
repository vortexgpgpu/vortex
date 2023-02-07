import uvm_pkg::*;
`include "uvm_macros.svh"

`include "ahb_bus_transaction_v2.svh"
`include "dut_params.svh"

class basic_sequence extends uvm_sequence #(ahb_bus_transaction_v2);
  `uvm_object_utils(basic_sequence)
  function new(string name = "");
    super.new(name);
  endfunction : new

  task body();
    ahb_bus_transaction_v2 req_item;
    req_item = ahb_bus_transaction_v2::type_id::create("req_item");

    // Do a simple NONSEQ or IDLE transfer
    repeat (10000) begin
      start_item(req_item);
      if (!req_item.randomize() with {
            req_item.hsize == 3'b010;
            foreach (req_item.haddr[i]) {
              req_item.haddr[i] < `AHB_NWORDS * 4;
              req_item.haddr[i][1:0] == 2'b00;
              //req_item.hwstrb == '1;
            }
            req_item.burstType == 3;
            req_item.errorAddrFlag == 0;
          }) begin
        // if the transaction is unable to be randomized, send a fatal message
        `uvm_fatal("sequence", "not able to randomize")
      end

      req_item.hsel = 1;

      finish_item(req_item);
    end
  endtask : body
endclass  //sequence

class burst_sequence extends uvm_sequence #(ahb_bus_transaction_v2);
  `uvm_object_utils(burst_sequence)
  function new(string name = "");
    super.new(name);
  endfunction : new

  task body();
    ahb_bus_transaction_v2 req_item;
    req_item = ahb_bus_transaction_v2::type_id::create("req_item");

    repeat (10000) begin
      start_item(req_item);
      if (!req_item.randomize() with {
            req_item.hsize == 3'b010;
            foreach (req_item.haddr[i]) {
              req_item.haddr[i] < `AHB_NWORDS * 4;
              req_item.haddr[i][1:0] == 2'b00;
              req_item.burstSize < 3;  // burstSize of 3 is invalid so needs to be less than
            }
            req_item.errorAddrFlag == 0;
          }) begin
        // if the transaction is unable to be randomized, send a fatal message
        `uvm_fatal("sequence", "not able to randomize")
      end

      req_item.hsel = 1;

      finish_item(req_item);
    end
  endtask : body
endclass  //sequence

class outOfBounds_sequence extends uvm_sequence #(ahb_bus_transaction_v2);
  `uvm_object_utils(outOfBounds_sequence)
  function new(string name = "");
    super.new(name);
  endfunction : new

  task body();
    ahb_bus_transaction_v2 req_item;
    req_item = ahb_bus_transaction_v2::type_id::create("req_item");

    repeat (1000) begin
      start_item(req_item);
      if (!req_item.randomize() with {
            req_item.hsize == 3'b010;
            foreach (req_item.haddr[i]) {
              req_item.haddr[i] >= `AHB_NWORDS * 4;
              req_item.haddr[i][1:0] == 2'b00;
              req_item.burstSize < 3;  // burstSize of 3 is invalid so needs to be less than
            }
            req_item.errorAddrFlag == 0;
            req_item.waitAddrFlag == 0;
          }) begin
        // if the transaction is unable to be randomized, send a fatal message
        `uvm_fatal("sequence", "not able to randomize")
      end

      req_item.hsel = 1;

      finish_item(req_item);
    end
  endtask : body
endclass  //sequence

class unaligned_sequence extends uvm_sequence #(ahb_bus_transaction_v2);
  `uvm_object_utils(unaligned_sequence)
  function new(string name = "");
    super.new(name);
  endfunction : new

  task body();
    ahb_bus_transaction_v2 req_item;
    req_item = ahb_bus_transaction_v2::type_id::create("req_item");

    repeat (1000) begin
      start_item(req_item);
      if (!req_item.randomize() with {
            req_item.hsize == 3'b010;
            foreach (req_item.haddr[i]) {
              req_item.haddr[i] < `AHB_NWORDS * 4;
              req_item.haddr[i][1:0] != 2'b00;
              req_item.burstSize < 3;  // burstSize of 3 is invalid so needs to be less than
            }
            req_item.errorAddrFlag == 0;
          }) begin
        // if the transaction is unable to be randomized, send a fatal message
        `uvm_fatal("sequence", "not able to randomize")
      end

      req_item.hsel = 1;

      finish_item(req_item);
    end
  endtask : body
endclass  //sequence

class errorAddr_sequence extends uvm_sequence #(ahb_bus_transaction_v2);
  `uvm_object_utils(errorAddr_sequence)
  function new(string name = "");
    super.new(name);
  endfunction : new

  task body();
    ahb_bus_transaction_v2 req_item;
    req_item = ahb_bus_transaction_v2::type_id::create("req_item");

    repeat (1000) begin
      start_item(req_item);
      if (!req_item.randomize() with {
            req_item.hsize == 3'b010;
            req_item.errorAddrFlag == 1;
            req_item.haddr[0][1:0] == 2'b00;
            req_item.burstSize < 3;  // burstSize of 3 is invalid so needs to be less than
            req_item.burstType == 3;
            req_item.idle != 1;
          }) begin
        // if the transaction is unable to be randomized, send a fatal message
        `uvm_fatal("sequence", "not able to randomize")
      end

      req_item.hsel = 1;

      finish_item(req_item);
    end
  endtask : body
endclass  //sequence

class waitAddr_sequence extends uvm_sequence #(ahb_bus_transaction_v2);
  `uvm_object_utils(waitAddr_sequence)
  function new(string name = "");
    super.new(name);
  endfunction : new

  task body();
    ahb_bus_transaction_v2 req_item;
    req_item = ahb_bus_transaction_v2::type_id::create("req_item");

    repeat (1000) begin
      start_item(req_item);
      if (!req_item.randomize() with {
            req_item.hsize == 3'b010;
            req_item.waitAddrFlag == 1;
            req_item.errorAddrFlag == 0;
            req_item.haddr[0][1:0] == 2'b00;
            req_item.burstSize < 3;  // burstSize of 3 is invalid so needs to be less than
            req_item.burstType == 3;
            req_item.idle != 1;
          }) begin
        // if the transaction is unable to be randomized, send a fatal message
        `uvm_fatal("sequence", "not able to randomize")
      end

      req_item.hsel = 1;

      finish_item(req_item);
    end
  endtask : body
endclass  //sequence

class sequencer extends uvm_sequencer #(ahb_bus_transaction_v2);
  `uvm_component_utils(sequencer)

  function new(input string name = "sequencer", uvm_component parent = null);
    super.new(name, parent);
  endfunction : new
endclass : sequencer
