import uvm_pkg::*;
`include "uvm_macros.svh"
`include "environment.svh"
`include "ahb_if.sv"

class testAll extends uvm_test;
  `uvm_component_utils(testAll)

  environment env;
  virtual ahb_if ahb_if;
  basic_sequence basicSeq;
  burst_sequence burstSeq;
  outOfBounds_sequence outOfBoundsSeq;
  errorAddr_sequence errorAddrSeq;
  waitAddr_sequence waitAddrSeq;
  unaligned_sequence unalignedSeq;

  function new(string name = "test", uvm_component parent);
    super.new(name, parent);
  endfunction : new

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = environment::type_id::create("env", this);
    basicSeq = basic_sequence::type_id::create("basicSeq");
    burstSeq = burst_sequence::type_id::create("burstSeq");
    outOfBoundsSeq = outOfBounds_sequence::type_id::create("outOfBoundsSeq");
    errorAddrSeq = errorAddr_sequence::type_id::create("errorAddrSeq");
    unalignedSeq = unaligned_sequence::type_id::create("unalignedSeq");
    waitAddrSeq = waitAddr_sequence::type_id::create("waitAddrSeq");


    // send the interface down
    if (!uvm_config_db#(virtual ahb_if)::get(this, "", "ahb_vif", ahb_if)) begin
      // check if interface is correctly set in testbench top level
      `uvm_fatal("TEST", "No virtual interface specified for this test instance")
    end

    uvm_config_db#(virtual ahb_if)::set(this, "env.agt*", "ahb_vif", ahb_if);

  endfunction : build_phase

  task run_phase(uvm_phase phase);
    phase.raise_objection(this, "Starting basic non seq sequence in main phase");
    `uvm_info(this.get_name(), "Starting basic sequence....", UVM_LOW);
    basicSeq.start(env.ahb_agent.sqr);
    `uvm_info(this.get_name(), "Finished basic sequence", UVM_LOW);
    #100ns;
    `uvm_info(this.get_name(), "Starting burst sequence....", UVM_LOW);
    burstSeq.start(env.ahb_agent.sqr);
    `uvm_info(this.get_name(), "Finished burst sequence", UVM_LOW);
    #100ns;
    `uvm_info(this.get_name(), "Starting addrOutOfBounds sequence....", UVM_LOW);
    outOfBoundsSeq.start(env.ahb_agent.sqr);
    `uvm_info(this.get_name(), "Finished addrOutOfBounds sequence", UVM_LOW);
    #100ns;
    `uvm_info(this.get_name(), "Starting errorAddress sequence....", UVM_LOW);
    //errorAddrSeq.start(env.ahb_agent.sqr);
    `uvm_info(this.get_name(), "Finished errorAddress sequence", UVM_LOW);
    #100ns;
    `uvm_info(this.get_name(), "Starting unalignedAddress sequence....", UVM_LOW);
    //unalignedSeq.start(env.ahb_agent.sqr);
    `uvm_info(this.get_name(), "Finished unalignedAddress sequence", UVM_LOW);
    #100ns;
    `uvm_info(this.get_name(), "Starting waitAddress sequence....", UVM_LOW);
    //waitAddrSeq.start(env.ahb_agent.sqr);
    `uvm_info(this.get_name(), "Finished waitAddress sequence", UVM_LOW);
    #100ns;
    phase.drop_objection(this, "Finished basic non seq sequence in main phase");
  endtask

endclass : testAll
