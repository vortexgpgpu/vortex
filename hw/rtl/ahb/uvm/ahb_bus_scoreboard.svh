`ifndef AHB_BUS_SCOREBOARD
`define AHB_BUS_SCOREBOARD

import uvm_pkg::*;
`include "uvm_macros.svh"

class ahb_bus_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(ahb_bus_scoreboard)
  uvm_analysis_export #(ahb_bus_transaction_v2) expected_export;  // receive result from predictor
  uvm_analysis_export #(ahb_bus_transaction_v2) actual_export;  // receive result from DUT
  uvm_tlm_analysis_fifo #(ahb_bus_transaction_v2) expected_fifo;
  uvm_tlm_analysis_fifo #(ahb_bus_transaction_v2) actual_fifo;

  int m_matches, m_mismatches;  // records number of matches and mismatches
  int numData;  // the number of data elements we should be looping over

  function new(string name, uvm_component parent);
    super.new(name, parent);
    m_matches = 0;
    m_mismatches = 0;
  endfunction

  function void build_phase(uvm_phase phase);
    expected_export = new("expected_export", this);
    actual_export = new("actual_export", this);
    expected_fifo = new("expected_fifo", this);
    actual_fifo = new("actual_fifo", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    expected_export.connect(expected_fifo.analysis_export);
    actual_export.connect(actual_fifo.analysis_export);
  endfunction

  task run_phase(uvm_phase phase);
    ahb_bus_transaction_v2 expected_tx;  //transaction from predictor
    ahb_bus_transaction_v2 actual_tx;  //transaction from DUT
    forever begin
      expected_fifo.get(expected_tx);
      actual_fifo.get(actual_tx);

      if (expected_tx.compare(actual_tx)) begin
        m_matches++;
        //uvm_report_info("Comparator", "Data Match");
      end else begin
        m_mismatches++;
        uvm_report_error("Comparator", "Error: Data Mismatch");
        if (actual_tx.burstType == 2) begin  // undefined length incrementing
          numData = actual_tx.burstLength;
        end else if (actual_tx.burstType == 3) begin
          numData = 1;
        end else begin
          case (actual_tx.burstSize)
            0: numData = 4;
            1: numData = 8;
            2: numData = 16;
          endcase
        end
        for (int i = 0; i < numData; i++) begin
          uvm_report_info("Comparator", $psprintf(
                          "\n\n0x%0h addr: 0x%0h --> expected:\nhrdata: 0x%0h\numErrors: 0x%0h\nhready_timeout: 0x%0h\n timeout_amount: 0x%0h\n~~~~~~~~~~~~~~~~~~\nactual:\nhrdata_out: 0x%0h\nnumErrors_out: 0x%0h\nhready_timeout_out: 0x%0h\n timeout_amount: 0x%0h",
                          i,
                          expected_tx.haddr[i],
                          expected_tx.hrdata_out[i],
                          expected_tx.numErrors[i],
                          expected_tx.hready_timeout[i],
                          expected_tx.timeout_amount[i],
                          actual_tx.hrdata_out[i],
                          actual_tx.numErrors[i],
                          actual_tx.hready_timeout[i],
                          actual_tx.timeout_amount[i]
                          ));
        end
      end
    end
  endtask

  function void report_phase(uvm_phase phase);
    uvm_report_info("Comparator", $sformatf("Matches:    %0d", m_matches));
    uvm_report_info("Comparator", $sformatf("Mismatches: %0d", m_mismatches));
  endfunction

endclass

`endif
