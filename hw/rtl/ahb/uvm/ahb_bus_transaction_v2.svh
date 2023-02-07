`ifndef TRANSACTION_SVH
`define TRANSACTION_SVH

import uvm_pkg::*;
`include "uvm_macros.svh"
`include "dut_params.svh"

class ahb_bus_transaction_v2 extends uvm_sequence_item;

  // Address phase items
  rand bit [127:0][`AHB_ADDR_WIDTH-1:0] haddr;
  rand bit [127:0][`AHB_DATA_WIDTH-1:0] hwdata;
  rand bit [127:0][(`AHB_DATA_WIDTH/8)-1:0] hwstrb;

  rand
  bit
  errorAddrFlag;  // 1 --> the address should be taken from the error array in dut_params.svh

  rand
  bit [31:0]
  errorAddrIndex;  // the index in the errorAddr array we should be taking the error address from 

  rand bit waitAddrFlag;  // 1 --> the address should be taken from the wait array in dut_params.svh

  rand
  bit [31:0]
  waitAddrIndex;  // the index in the waitAddr array we should be taking the wait address from 

  rand bit rw;  // o --> read; 1 --> write
  rand bit idle;  // 0 --> not idle transfer, 1 --> idle transfer

  rand
  bit [1:0]
  burstType; // 0 --> wrapping, 1 --> incrementing, 2 --> undefined length incrementing, 3 --> single transfer

  rand
  bit [6:0]
  burstLength; // For an undefined length incrementing burst transfer we actually define the length within the testbench

  rand bit [1:0] burstSize;  // 0 --> 4, 1 --> 8, 2 --> 16

  rand bit [1:0] hsize;  // 0 --> byte transfer, 1 --> halfword transfer, 2 --> word transfer
  bit hsel; // This indicates whether we should even select this subordinate regardless of the rest of the transfer


  // Data phase items
  bit [127:0][`AHB_ADDR_WIDTH-1:0] hrdata_out;
  //bit [125:0] [1:0] hresp_out; // two bits, one for the first error slot and one for the second error slot
  logic [127:0][31:0] numErrors;  // indiactes the number of error states seen for each transaction
  bit [127:0] hready_timeout;  // indicated that hready had a timeout

  logic [127:0][31:0] timeout_amount;  // indiactes the number of wait states seen for each transaction
  constraint bursts {
    burstType < 4;
    burstSize < 4;
  }
  ;
  constraint sizes {hsize < 3;}
  ;

  constraint errorAddr {errorAddrIndex < `AHB_NUM_ERROR_ADDR;}
  ;

  constraint waitAddr {waitAddrIndex < `AHB_NUM_WAIT_ADDR;}
  ;

  //TODO: YOU MAY WANT TO RECONSIDER HOW MANY OF THESE FIELDS YOU INCLUDE FOR PRINTING
  // NOTE: EXAMPLE OF NOT PRINTING CERTAIN FIELDS BELOW
  // `uvm_field_int(haddr, UVM_NOCOMPARE | UVM_NOPRINT)

  `uvm_object_utils_begin(ahb_bus_transaction_v2)
    `uvm_field_int(errorAddrIndex, UVM_NOCOMPARE)
    `uvm_field_int(errorAddrFlag, UVM_NOCOMPARE)
    `uvm_field_int(waitAddrFlag, UVM_NOCOMPARE)
    `uvm_field_int(waitAddrIndex, UVM_NOCOMPARE)
    `uvm_field_int(haddr, UVM_NOCOMPARE)
    `uvm_field_sarray_int(hwdata, UVM_NOCOMPARE)
    `uvm_field_sarray_int(hwstrb, UVM_NOCOMPARE)
    `uvm_field_int(rw, UVM_NOCOMPARE)
    `uvm_field_int(idle, UVM_NOCOMPARE)
    `uvm_field_int(burstType, UVM_NOCOMPARE)
    `uvm_field_int(burstLength, UVM_NOCOMPARE)
    `uvm_field_int(burstSize, UVM_NOCOMPARE)
    `uvm_field_int(hsel, UVM_NOCOMPARE)
    `uvm_field_int(hsize, UVM_NOCOMPARE)
  // `uvm_field_int(hresp_out, UVM_DEFAULT)
    `uvm_field_int(hrdata_out, UVM_DEFAULT)
    `uvm_field_int(hready_timeout, UVM_DEFAULT)
    `uvm_field_int(numErrors, UVM_DEFAULT)
    `uvm_field_int(timeout_amount, UVM_DEFAULT)
  `uvm_object_utils_end

  function new(string name = "ahb_bus_transaction_v2");
    super.new(name);
  endfunction : new

endclass : ahb_bus_transaction_v2

`endif
