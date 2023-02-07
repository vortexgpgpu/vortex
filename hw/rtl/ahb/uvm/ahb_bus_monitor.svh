`ifndef AHB_BUS_MONITOR_SVH
`define AHB_BUS_MONITOR_SVH

import uvm_pkg::*;
`include "uvm_macros.svh"
`include "ahb_if.sv"
class ahb_bus_monitor extends uvm_monitor;
  `uvm_component_utils(ahb_bus_monitor)

  virtual ahb_if vif;

  uvm_analysis_port #(ahb_bus_transaction_v2) ahb_bus_ap;
  uvm_analysis_port #(ahb_bus_transaction_v2) result_ap;
  int timeoutCount;

  function new(string name, uvm_component parent = null);
    super.new(name, parent);
    ahb_bus_ap = new("ahb_bus_ap", this);
    result_ap  = new("result_ap", this);
  endfunction : new

  // Build Phase - Get handle to virtual if from config_db
  virtual function void build_phase(uvm_phase phase);
    if (!uvm_config_db#(virtual ahb_if)::get(this, "", "ahb_vif", vif)) begin
      `uvm_fatal("monitor", "No virtual interface specified for this monitor instance")
    end
  endfunction

  virtual task run_phase(uvm_phase phase);
    super.run_phase(phase);
    @(posedge vif.HCLK);
    forever begin
      ahb_bus_transaction_v2 tx;
      timeoutCount = 0;
      // captures activity between the driver and DUT
      tx = ahb_bus_transaction_v2::type_id::create("tx");

      // zero out everything
      tx.haddr = '0;
      tx.hwdata = '0;
      tx.hwstrb = '0;
      tx.hrdata_out = '0;
      for (int i = 0; i < 126; i++) begin
        tx.numErrors[i] = 0;
      end
      for (int i = 0; i < 126; i++) begin
        tx.timeout_amount[i] = 0;
      end
      tx.hready_timeout = '0;

      tx.haddr[0] = vif.HADDR;
      //tx.hwdata[0] = vif.HWDATA;
      //tx.hwstrb[0] = vif.HWSTRB;
      tx.rw = vif.HWRITE;
      tx.idle = vif.HTRANS == 2'b0;

      case (vif.HBURST)
        0: tx.burstType = 3;  // single transfer
        1: tx.burstType = 2;  // undefined length incr
        2: tx.burstType = 0;  // wrap
        3: tx.burstType = 1;  // incr
        4: tx.burstType = 0;  // wrap
        5: tx.burstType = 1;  // incr
        6: tx.burstType = 0;  // wrap
        7: tx.burstType = 1;  // incr
      endcase

      case (vif.HBURST)
        2: tx.burstSize = 0;  // size 4
        3: tx.burstSize = 0;
        4: tx.burstSize = 1;  // size 8
        5: tx.burstSize = 1;
        6: tx.burstSize = 2;  // size 16
        7: tx.burstSize = 2;
        default: tx.burstSize = 0;
      endcase

      tx.hsize = vif.HSIZE;
      tx.hsel  = vif.HSEL;

      // check if there is a new input to the subordinate
      if (vif.HSEL == 1'b1) begin
        // send the new input data to predictor though ahb_bus_ap

        //`uvm_info(this.get_name(), "New item sent to predictor", UVM_LOW);
        // depending on what type of transaction
        case (vif.HTRANS)
          0: begin  // if it is an idle transfer
            @(posedge vif.HCLK);
            tx.hwdata[0] = vif.HWDATA;
            tx.hwstrb[0] = vif.HWSTRB;
            // We should see a zero wait state okay response
            while (!vif.HREADYOUT && timeoutCount != 1 && !vif.HRESP) begin
              timeoutCount = timeoutCount + 1;
              @(posedge vif.HCLK);
            end
            //tx.hrdata_out[0]  = vif.HRDATA; don't worry about data, master won't be reading it during idle anyways
            //tx.hresp_out[0][0] = vif.HRESP;
            tx.numErrors[0]   = vif.HRESP;  // if there is an error this will count one
            tx.hready_timeout = timeoutCount == 1;
            tx.timeout_amount = timeoutCount;
          end
          2: begin  // NONSEQ transfers
            if (vif.HBURST == '0) begin  // single transfer
              @(posedge vif.HCLK);
              tx.hwdata[0] = vif.HWDATA;
              tx.hwstrb[0] = vif.HWSTRB;
              while (!vif.HREADYOUT && timeoutCount != 100 && !vif.HRESP) begin
                timeoutCount = timeoutCount + 1;
                @(posedge vif.HCLK);
              end
              tx.timeout_amount = timeoutCount;
              tx.hready_timeout = timeoutCount == 100;
              if (!tx.rw) begin  // if we're not writing, we care about the hrdata outptu
                tx.hrdata_out[0] = vif.HRDATA;
              end
              //tx.hresp_out[0][0] = vif.HRESP;
              tx.numErrors[0] = 0;
              while (vif.HRESP && timeoutCount == 0) begin  // if we see an error state we should count them all
                tx.numErrors[0] = tx.numErrors[0] + 1;
                @(posedge vif.HCLK);
              end
            end else begin  // All other burst types look the same basically
              tx.burstLength = 1;
              for (int i = 0; i < 128; i++) begin  // the max length of undefined burst is 128
                //`uvm_info(this.get_name(), "Monitor burst", UVM_LOW);
                timeoutCount = 0;
                @(posedge vif.HCLK);
                tx.hwdata[i] = vif.HWDATA;
                tx.hwstrb[i] = vif.HWSTRB;
                while (!vif.HREADYOUT && timeoutCount != 100 && !vif.HRESP) begin
                  timeoutCount = timeoutCount + 1;
                  @(posedge vif.HCLK);
                end
                tx.timeout_amount[i] = timeoutCount;

                tx.hready_timeout = timeoutCount == 100;
                if (!tx.rw) begin  // if we are writing, we don't care about hrdata output
                  tx.hrdata_out[i] = vif.HRDATA;
                end
                //tx.hresp_out[0][0] = vif.HRESP;
                tx.numErrors[i] = 0;
                while (vif.HRESP && timeoutCount == 0) begin  // if we see an error state we should count them all
                  tx.numErrors[i] = tx.numErrors[i] + 1;
                  @(posedge vif.HCLK);
                end

                if(vif.HTRANS == 2'b0 || vif.HTRANS == 2'b10 || i == 127) begin // break out if this is the end of the transfer
                  break;
                end
                // Set the next transactions signals
                tx.haddr[i+1]  = vif.HADDR;
                //tx.hwdata[i+1] = vif.HWDATA;
                //tx.hwstrb[i+1] = vif.HWSTRB;
                tx.burstLength = tx.burstLength + 1;
              end
            end
          end
        endcase

        ahb_bus_ap.write(tx);

        // now write the result to the scoreboard!
        //`uvm_info(this.get_name(), "New result sent to scoreboard", UVM_LOW);
        result_ap.write(tx);
      end else begin
        @(posedge vif.HCLK);
      end

    end
  endtask : run_phase

endclass : ahb_bus_monitor

`endif
