`ifndef AHB_BUS_DRIVER_SVH
`define AHB_BUS_DRIVER_SVH

import uvm_pkg::*;
`include "uvm_macros.svh"

`include "ahb_if.sv"

class ahb_bus_driver extends uvm_driver #(ahb_bus_transaction_v2);
  `uvm_component_utils(ahb_bus_driver)

  virtual ahb_if vif;
  int timeoutCount;
  logic dPhaseRdy, aPhaseRdy;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction : new

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    // get interface from database
    if (!uvm_config_db#(virtual ahb_if)::get(this, "", "ahb_vif", vif)) begin
      // if the interface was not correctly set, raise a fatal message
      `uvm_fatal("Driver", "No virtual interface specified for this test instance");
    end
  endfunction : build_phase

  task run_phase(uvm_phase phase);
    ahb_bus_transaction_v2 currTransaction;
    ahb_bus_transaction_v2 prevTransaction;
    logic firstFlag;
    int burstCount;
    int prevTransIndex;
    int currTransIndex;
    logic errorFlag;
    int actualBurstLength;

    DUT_reset();  // Power on Reset

    firstFlag = 1;
    prevTransaction = ahb_bus_transaction_v2::type_id::create("prevTransaction");

    forever begin  // address  phase block
      @(negedge vif.HCLK);
      if (firstFlag) begin  // if this is our very first time through the forever loop
        seq_item_port.get_next_item(currTransaction);
        /*`uvm_info(this.get_name(), $sformatf("Received new sequence item:\n%s",
                                             currTransaction.sprint()), UVM_LOW);*/
        //`uvm_info(this.get_name(), "Received new sequence item", UVM_LOW);

        prevTransaction.idle = 1;
      end else begin
        zero_sigs();
        prevTransaction.copy(currTransaction);
        seq_item_port.item_done();
        //`uvm_info(this.get_name(), "single transfer complete", UVM_LOW);

        seq_item_port.get_next_item(currTransaction);

        prevTransIndex = currTransIndex;
        /*`uvm_info(this.get_name(), $sformatf("Received new sequence item:\n%s",
                                             currTransaction.sprint()), UVM_LOW);*/
        //`uvm_info(this.get_name(), "Received new sequence item", UVM_LOW);
      end


      //Addr0 drive
      driveSignals(0, currTransaction, 0, burstCount);
      driveWData(prevTransIndex, prevTransaction);

      currTransIndex = 0;

      @(posedge vif.HCLK);

      if (~firstFlag) begin
        // DATAN sample by manager
        timeoutCount = 0;
        while (!vif.HREADYOUT && !vif.HRESP && timeoutCount != 1000) begin
          //`uvm_info(this.get_name(), "waited for ready signal", UVM_LOW);
          timeoutCount = timeoutCount + 1;
          @(posedge vif.HCLK);  // wait for the subordinate to signal that it is ready
        end

        if (timeoutCount == 1000) begin
          `uvm_fatal(this.get_name(), "Timeout while waiting for subordinate ready response");
        end

        if (vif.HRESP && !vif.HREADYOUT) begin // we have encountered an error condition, handle appropriately

          // Inserting an idle state
          @(negedge vif.HCLK);
          vif.HTRANS = '0;
          vif.HBURST = '0;
          vif.HADDR  = '0;
          @(posedge vif.HCLK);
          @(negedge vif.HCLK);

          // Now redrive the signals
          driveSignals(0, currTransaction, 0, burstCount);
          driveWData(prevTransIndex, prevTransaction);
        end
      end else begin
        firstFlag = 0;
      end

      // determine the length of the burst we are dealing with
      if (currTransaction.burstType == 3 || currTransaction.idle) begin  // single transfer or idle
        actualBurstLength = 0;
      end else if (currTransaction.burstType == 2) begin  // undefined length burst
        burstCount = burstCount + 1;
        actualBurstLength = currTransaction.burstLength;  // defined by testbench!
      end else begin  // incrementing burst or wrapping burst
        burstCount = burstCount + 1;
        case (currTransaction.burstSize)
          0: actualBurstLength = 4;
          1: actualBurstLength = 8;
          2: actualBurstLength = 16;
        endcase
      end

      for (int i = 1; i < actualBurstLength; i++) begin  // loop through the entire burst
        //`uvm_info(this.get_name(), "looping1", UVM_LOW);
        // Next burst address phase
        @(negedge vif.HCLK);
        //`uvm_info(this.get_name(), "looping2", UVM_LOW);
        driveSignals(i, currTransaction, 1, burstCount - 1);
        driveWData(i - 1, currTransaction);

        // Now do the previous indexs data phase
        @(posedge vif.HCLK);
        while (!vif.HREADYOUT && !vif.HRESP && timeoutCount != 1000) begin
          //`uvm_info(this.get_name(), "waited for ready signal", UVM_LOW);
          timeoutCount = timeoutCount + 1;
          @(posedge vif.HCLK);  // wait for the subordinate to signal that it is ready
        end

        if (timeoutCount == 1000) begin
          `uvm_fatal(this.get_name(), "Timeout while waiting for subordinate ready response");
        end

        if (vif.HRESP && !vif.HREADYOUT) begin // we have encountered an error condition, handle appropriately
          // Inserting an idle state
          @(negedge vif.HCLK);
          vif.HTRANS = '0;
          vif.HBURST = '0;
          vif.HADDR  = '0;
          @(posedge vif.HCLK);
          @(negedge vif.HCLK);

          // Now redrive the signals
          driveSignals(i, currTransaction, 1, burstCount - 1);
          driveWData(i - 1, currTransaction);
        end

      end

    end
  endtask



  task driveWData(input int currIndex, input ahb_bus_transaction_v2 currTransaction);

    vif.HWDATA = currTransaction.hwdata[currIndex];
    vif.HWSTRB = currTransaction.hwstrb[currIndex];
  endtask

  task driveSignals(input int currIndex, input ahb_bus_transaction_v2 currTransaction,
                    input logic sequential, input int burstIndex);
    int numBeats;
    logic [31:0] addrTop;
    if (currTransaction.idle) begin
      vif.HTRANS = '0;
      vif.HSEL   = currTransaction.hsel;
      //vif.HWDATA = currTransaction.hwdata[currIndex];
      vif.HADDR  = '0;
      //vif.HWSTRB = currTransaction.hwstrb[currIndex];
      vif.HBURST = '0;
      vif.HSIZE  = currTransaction.hsize;

    end else if (currTransaction.burstType == 3) begin  // Single transfer!
      vif.HTRANS = 2'b10;  // NONSEQ transfer
      vif.HSEL   = currTransaction.hsel;
      vif.HWRITE = currTransaction.rw;
      vif.HBURST = '0;
      vif.HSIZE  = currTransaction.hsize;
      if (currTransaction.errorAddrFlag) begin
        vif.HADDR = errorAddr[currTransaction.errorAddrIndex];
      end else if (currTransaction.waitAddrFlag) begin
        vif.HADDR = waitAddr[currTransaction.waitAddrIndex];
      end else begin
        vif.HADDR = currTransaction.haddr[currIndex];
      end
      //vif.HWDATA = currTransaction.hwdata[currIndex];
      //vif.HWSTRB = currTransaction.hwstrb[currIndex];
    end else if (currTransaction.burstType == 1) begin  // incrementing burst transfer
      vif.HTRANS = sequential ? 2'b11 : 2'b10;  // NONSEQ  or SEQ transfer depending
      vif.HSEL   = currTransaction.hsel;
      vif.HWRITE = currTransaction.rw;
      vif.HSIZE  = currTransaction.hsize;
      //vif.HWDATA = currTransaction.hwdata[currIndex];
      //vif.HWSTRB = currTransaction.hwstrb[currIndex];

      case (currTransaction.burstSize)
        0: vif.HBURST = 3'b011;  // INCR4
        1: vif.HBURST = 3'b101;  // INCR8
        2: vif.HBURST = 3'b111;  // INCR16
      endcase

      if (currTransaction.errorAddrFlag) begin
        vif.HADDR = errorAddr[currTransaction.errorAddrIndex] + currIndex * currTransaction.hsize * 2;
      end else if (currTransaction.waitAddrFlag) begin
        vif.HADDR = waitAddr[currTransaction.waitAddrIndex] + currIndex * currTransaction.hsize * 2;
      end else begin
        vif.HADDR = currTransaction.haddr[burstIndex] + currIndex * currTransaction.hsize * 2;
      end

    end else if (currTransaction.burstType == 0) begin  // wrapping burst transfer
      vif.HTRANS = sequential ? 2'b11 : 2'b10;  // NONSEQ  or SEQ transfer depending
      vif.HSEL   = currTransaction.hsel;
      vif.HWRITE = currTransaction.rw;
      vif.HSIZE  = currTransaction.hsize;
      //vif.HWDATA = currTransaction.hwdata[currIndex];
      //vif.HWSTRB = currTransaction.hwstrb[currIndex];

      case (currTransaction.burstSize)
        0: begin
          vif.HBURST = 3'b010;  // WRAP4
          numBeats   = 4;
        end
        1: begin
          vif.HBURST = 3'b100;  // WRAP8
          numBeats   = 8;
        end
        2: begin
          vif.HBURST = 3'b110;  // WRAP16
          numBeats   = 16;
        end
      endcase

      if (currTransaction.errorAddrFlag) begin
        // The wrapping address doesn't change the top portion of the address so we need to zero out
        // the bottom portion that wraps around so we can then set it in the next statement
        addrTop = (errorAddr[currTransaction.errorAddrIndex] >>
                   $clog2(2 * currTransaction.hsize * numBeats)) <<
            $clog2(2 * currTransaction.hsize * numBeats);

        vif.HADDR  = addrTop + (errorAddr[currTransaction.errorAddrIndex] + (currIndex * currTransaction.hsize * 2)) % (2 * currTransaction.hsize * numBeats);
      end else if (currTransaction.waitAddrFlag) begin
        // The wrapping address doesn't change the top portion of the address so we need to zero out
        // the bottom portion that wraps around so we can then set it in the next statement
        addrTop = (waitAddr[currTransaction.waitAddrIndex] >>
                   $clog2(2 * currTransaction.hsize * numBeats)) <<
            $clog2(2 * currTransaction.hsize * numBeats);

        vif.HADDR  = addrTop + (waitAddr[currTransaction.waitAddrIndex] + (currIndex * currTransaction.hsize * 2)) % (2 * currTransaction.hsize * numBeats);
      end else begin
        addrTop = (currTransaction.haddr[burstIndex] >> $clog2(2 * currTransaction.hsize * numBeats)
            ) << $clog2(2 * currTransaction.hsize * numBeats);

        vif.HADDR  = addrTop + (currTransaction.haddr[burstIndex] + (currIndex * currTransaction.hsize * 2)) % (2 * currTransaction.hsize * numBeats);
      end


    end else if (currTransaction.burstType == 2) begin  // undefined length burst
      vif.HTRANS = sequential ? 2'b11 : 2'b10;  // NONSEQ  or SEQ transfer depending
      vif.HSEL   = currTransaction.hsel;
      vif.HWRITE = currTransaction.rw;
      vif.HSIZE  = currTransaction.hsize;

      //vif.HWDATA = currTransaction.hwdata[currIndex];
      //vif.HWSTRB = currTransaction.hwstrb[currIndex];
      vif.HBURST = 3'b001;

      if (currTransaction.errorAddrFlag) begin
        vif.HADDR = errorAddr[currTransaction.errorAddrIndex] + currIndex * currTransaction.hsize * 2;
      end else if (currTransaction.waitAddrFlag) begin
        vif.HADDR = waitAddr[currTransaction.waitAddrIndex] + currIndex * currTransaction.hsize * 2;
      end else begin
        vif.HADDR = currTransaction.haddr[burstIndex] + currIndex * currTransaction.hsize * 2;
      end

    end

  endtask

  task DUT_reset();
    `uvm_info(this.get_name(), "Resetting DUT", UVM_LOW);

    vif.HSEL    = '0;
    vif.HWRITE  = '0;
    vif.HTRANS  = '0;
    vif.HADDR   = '0;
    vif.HWDATA  = '0;
    vif.HRESETn = '1;

    @(posedge vif.HCLK);
    vif.HRESETn = '0;
    @(posedge vif.HCLK);
    vif.HRESETn = '1;
    @(posedge vif.HCLK);
    @(posedge vif.HCLK);
  endtask

  task zero_sigs();
    vif.HTRANS = '0;
    vif.HSEL   = '0;
    vif.HWRITE = '0;
    vif.HBURST = '0;
    vif.HSIZE  = '0;
    vif.HADDR  = '0;
    vif.HWDATA = '0;
    vif.HWSTRB = '0;
  endtask

endclass : ahb_bus_driver

`endif
