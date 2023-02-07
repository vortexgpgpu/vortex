`ifndef AHB_BUS_PREDICTOR
`define AHB_BUS_PREDICTOR

import uvm_pkg::*;
`include "uvm_macros.svh"
`include "ahb_bus_transaction_v2.svh"
`include "peripheral_model.svh"
`include "dut_params.svh"

localparam TOP_ADDR = `AHB_BASE_ADDR + `AHB_NWORDS * (`AHB_ADDR_WIDTH / 8);

class ahb_bus_predictor extends uvm_subscriber #(ahb_bus_transaction_v2);
  `uvm_component_utils(ahb_bus_predictor)

  peripheral_model periph = new;

  uvm_analysis_port #(ahb_bus_transaction_v2) pred_ap;
  ahb_bus_transaction_v2 output_tx;

  function new(string name, uvm_component parent = null);
    super.new(name, parent);
  endfunction : new

  function void build_phase(uvm_phase phase);
    pred_ap = new("pred_ap", this);
  endfunction


  function void write(ahb_bus_transaction_v2 t);

    integer burstLength;
    // t is the transaction sent from monitor
    output_tx = ahb_bus_transaction_v2::type_id::create("output_tx", this);
    output_tx.copy(t);

    output_tx.hrdata_out = '0;
    for (int i = 0; i < 126; i++) begin
      output_tx.numErrors[i] = 0;
      output_tx.timeout_amount[i] = 0;
    end
    output_tx.hready_timeout = '0;

    case (output_tx.burstType)
      0: begin  // wrapping
        if (output_tx.burstSize == 0) burstLength = 4;
        else if (output_tx.burstSize == 1) burstLength = 8;
        else if (output_tx.burstSize == 2) burstLength = 16;

        for (int i = 0; i < burstLength; i++) begin
          if((output_tx.haddr[i] < `AHB_BASE_ADDR || output_tx.haddr[i] >= TOP_ADDR) || (output_tx.haddr[i] & (`AHB_ADDR_WIDTH/8 - 1)) != 'b0) begin
            output_tx.numErrors[i] = 2;
          end
          for (int j = 0; j < `AHB_NUM_ERROR_ADDR; j++) begin
            if (errorAddr[j] == output_tx.haddr[i]) begin
              output_tx.numErrors[i] = 2;
            end
          end
          for (int j = 0; j < `AHB_NUM_WAIT_ADDR; j++) begin
            if (waitAddr[j] == output_tx.haddr[i] && output_tx.numErrors[i] == 0) begin
              output_tx.timeout_amount[i] = 1;
            end
          end
          if (output_tx.rw == 1 && output_tx.numErrors[i] == 0) begin  // writing
            periph.writeWord(output_tx.hwdata[i], output_tx.hwstrb[i], output_tx.haddr[i]);
          end else if (output_tx.numErrors[i] == 0) begin  // reading
            output_tx.hrdata_out[i] = periph.readWord(output_tx.haddr[i]);
          end
          output_tx.hready_timeout[i] = 0;
        end
      end
      1: begin  // incrementing
        if (output_tx.burstSize == 0) burstLength = 4;
        else if (output_tx.burstSize == 1) burstLength = 8;
        else if (output_tx.burstSize == 2) burstLength = 16;

        for (int i = 0; i < burstLength; i++) begin
          if((output_tx.haddr[i] < `AHB_BASE_ADDR || output_tx.haddr[i] >= TOP_ADDR) || (output_tx.haddr[i] & (`AHB_ADDR_WIDTH/8 - 1)) != 'b0) begin
            output_tx.numErrors[i] = 2;
          end
          for (int j = 0; j < `AHB_NUM_ERROR_ADDR; j++) begin
            if (errorAddr[j] == output_tx.haddr[i]) begin
              output_tx.numErrors[i] = 2;
            end
          end
          for (int j = 0; j < `AHB_NUM_WAIT_ADDR; j++) begin
            if (waitAddr[j] == output_tx.haddr[i] && output_tx.numErrors[i] == 0) begin
              output_tx.timeout_amount[i] = 1;
            end
          end
          if (output_tx.rw == 1 && output_tx.numErrors[i] == 0) begin  // writing
            periph.writeWord(output_tx.hwdata[i], output_tx.hwstrb[i], output_tx.haddr[i]);
          end else if (output_tx.numErrors[i] == 0) begin  // reading
            output_tx.hrdata_out[i] = periph.readWord(output_tx.haddr[i]);
          end
          output_tx.hready_timeout[i] = 0;
        end

      end
      2: begin  // undefined length incrementing
        //`uvm_info(this.get_name(), "Undefined increment", UVM_LOW);
        burstLength = output_tx.burstLength;
        //`uvm_info(this.get_name(), $psprintf("Burst length: %0d", burstLength), UVM_LOW);

        for (int i = 0; i < burstLength; i++) begin
          //`uvm_info(this.get_name(), "Undefined increment loop", UVM_LOW);
          if((output_tx.haddr[i] < `AHB_BASE_ADDR || output_tx.haddr[i] >= TOP_ADDR) || (output_tx.haddr[i] & (`AHB_ADDR_WIDTH/8 - 1)) != 'b0) begin
            output_tx.numErrors[i] = 2;
          end
          for (int j = 0; j < `AHB_NUM_ERROR_ADDR; j++) begin
            if (errorAddr[j] == output_tx.haddr[i]) begin
              output_tx.numErrors[i] = 2;
            end
          end
          for (int j = 0; j < `AHB_NUM_WAIT_ADDR; j++) begin
            if (waitAddr[j] == output_tx.haddr[i] && output_tx.numErrors[i] == 0) begin
              output_tx.timeout_amount[i] = 1;
            end
          end
          if (output_tx.rw == 1 && output_tx.numErrors[i] == 0) begin  // writing
            periph.writeWord(output_tx.hwdata[i], output_tx.hwstrb[i], output_tx.haddr[i]);
          end else if (output_tx.numErrors[i] == 0) begin  // reading
            output_tx.hrdata_out[i] = periph.readWord(output_tx.haddr[i]);
          end
          output_tx.hready_timeout[i] = 0;
        end
      end
      3: begin  // single transfer
        if((output_tx.haddr[0] < `AHB_BASE_ADDR || output_tx.haddr[0] >= TOP_ADDR) || (output_tx.haddr[0] & (`AHB_ADDR_WIDTH/8 - 1)) != 'b0) begin
          output_tx.numErrors[0] = 2;
        end
        for (int j = 0; j < `AHB_NUM_ERROR_ADDR; j++) begin
          if (errorAddr[j] == output_tx.haddr[0]) begin
            output_tx.numErrors[0] = 2;
          end
        end
        for (int j = 0; j < `AHB_NUM_WAIT_ADDR; j++) begin
          if (waitAddr[j] == output_tx.haddr[0] && output_tx.numErrors[0] == 0) begin
            output_tx.timeout_amount[0] = 1;
          end
        end
        if (output_tx.rw == 1 && output_tx.numErrors[0] == 0) begin  // writing
          periph.writeWord(output_tx.hwdata[0], output_tx.hwstrb[0], output_tx.haddr[0]);
        end else if (output_tx.numErrors[0] == 0) begin  // reading
          output_tx.hrdata_out[0] = periph.readWord(output_tx.haddr[0]);
        end
        output_tx.hready_timeout[0] = 0;
      end
      default: begin
        output_tx.hready_timeout[0] = 0;
        output_tx.hrdata_out[0] = '0;
        output_tx.numErrors[0] = 0;
        output_tx.timeout_amount[0] = 0;
      end
    endcase

    if (output_tx.idle) begin
      output_tx.hready_timeout[0] = 0;
      output_tx.hrdata_out[0] = '0;
      output_tx.numErrors[0] = 0;
      output_tx.timeout_amount[0] = 0;
    end

    // after prediction, the expected output send to the scoreboard 
    pred_ap.write(output_tx);
  endfunction : write

endclass : ahb_bus_predictor

`endif
