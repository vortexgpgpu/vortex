
`define FORMAT_ERROR(signal, expected, actual) \
    $display("In block %s test %d (%t)\n%s: Expected %h, got %h", test_block, test_num, $time(), signal, expected, actual);

module tb_ahbl_bus_mux;

    import ahb_pkg::*;

    logic HCLK = 0, HRESETn;
    localparam READ = 'b0;
    localparam WRITE = 'b1;
    parameter NMANAGERS = 3;

    always #(10) HCLK++;

    int test_num = -1;
    int transaction_num = 0;
    string test_block = "Initializing...";

    ahb_if m_in[NMANAGERS-1:0](HCLK, HRESETn);
    ahb_if m_out(HCLK, HRESETn);
    bus_protocol_if prif();

    //assign m_out.HREADY = m_out.HREADYOUT; // passthrough
    
    /* 
    * Arrays of interfaces don't support array indexing with variables 
    * -- cannot make generic tasks passing # of master to check. 
    *  Add indirection through structs 
    */
    typedef struct packed {
        logic [31:0] HADDR;
        logic [31:0] HWDATA;
        logic [2:0]  HSIZE;
        logic [1:0]  HTRANS;
        logic        HWRITE;
        logic [2:0]  HBURST;
        logic        HMASTLOCK;
        logic [3:0]  HWSTRB;
    } ahb_in_t;

    typedef struct packed {
        logic [31:0] HRDATA;
        logic [1:0]  HRESP;
        logic        HREADY;
    } ahb_out_t;

    ahb_in_t [NMANAGERS-1:0] master_inputs;
    ahb_out_t [NMANAGERS-1:0] master_outputs;
    int transactions [NMANAGERS-1:0];

    genvar i;
    generate
        for(i = 0; i < NMANAGERS; i++) begin
            assign m_in[i].HADDR = master_inputs[i].HADDR;
            assign m_in[i].HWDATA = master_inputs[i].HWDATA;
            assign m_in[i].HSIZE = master_inputs[i].HSIZE;
            assign m_in[i].HTRANS = master_inputs[i].HTRANS;
            assign m_in[i].HWRITE = master_inputs[i].HWRITE;
            assign m_in[i].HBURST = master_inputs[i].HBURST;
            assign m_in[i].HMASTLOCK = master_inputs[i].HMASTLOCK;
            assign m_in[i].HWSTRB = master_inputs[i].HWSTRB;

            assign master_outputs[i].HRDATA = m_in[i].HRDATA;
            assign master_outputs[i].HRESP = m_in[i].HRESP;
            assign master_outputs[i].HREADY = m_in[i].HREADYOUT;
        end
    endgenerate

    ahb_mux #(.NMANAGERS(NMANAGERS)) DUT(
        .HCLK,
        .HRESETn,
        .m_in,
        .m_out
    );
    
    ahb_subordinate #(
        .BASE_ADDR(0),
        .NWORDS(32)
    ) AHB_MEM(
        .ahb_if(m_out),
        .bus_if(prif)
    );

    simple_memory #(
           
    ) MEM(
        .CLK(HCLK),
        .latency(0),
        .prif,
        .hintif(prif)
    );

    task reset();
        @(negedge HCLK);
        HRESETn = '0;
        repeat(2) @(posedge HCLK);
        @(negedge HCLK);
        HRESETn = '1;
    endtask

    task clear_request(int n);
        master_inputs[n].HADDR = '0;
        master_inputs[n].HSIZE = '0;
        master_inputs[n].HTRANS = IDLE;
        master_inputs[n].HWRITE = '0;
        master_inputs[n].HBURST = '0;
        //master_inputs[n].HWDATA = '0;
        master_inputs[n].HMASTLOCK = '0;
    endtask
    
    task generate_request(int n, input logic HWRITE);
        transaction_num++;
        transactions[n] = transaction_num; // Record so we can check things
        master_inputs[n].HADDR = transaction_num << 2;
        master_inputs[n].HSIZE = 'h2;
        master_inputs[n].HTRANS = NONSEQ;
        master_inputs[n].HWRITE = HWRITE;
        master_inputs[n].HBURST = '0;
        //master_inputs[n].HWDATA = transaction_num; Need to assign this in dphase
        master_inputs[n].HMASTLOCK = '0;
    endtask

    task check_addr_phase(int nth_through);
        assert(m_out.HADDR == master_inputs[nth_through].HADDR)
        else `FORMAT_ERROR("HADDR", master_inputs[nth_through].HADDR, m_out.HADDR);
    endtask

    initial begin
        HRESETn = '1;
        clear_request(0);
        clear_request(1);
        clear_request(2);

        reset();
        @(posedge HCLK) #(1);
        m_out.HSEL = 1'b1;
        generate_request(0, WRITE);
        generate_request(1, WRITE);
        generate_request(2, WRITE);
        @(posedge HCLK) #(1);
        master_inputs[2].HWDATA = 32'hDEAD;//transactions[2]; // master 2 enters dphase
        master_inputs[2].HWSTRB = 4'hF;
        clear_request(2);
        @(posedge HCLK) #(1);
        master_inputs[1].HWDATA = 32'hBEEF;//transactions[1]; // master 2 enters dphase
        master_inputs[1].HWSTRB = 4'hF;
        clear_request(1);
        @(posedge HCLK) #(1);
        master_inputs[0].HWDATA = 32'hCAFE;//transactions[0]; // master 2 enters dphase
        master_inputs[0].HWSTRB = 4'hF;
        clear_request(0);
        m_out.HSEL = 1'b0;

        repeat(10) @(posedge HCLK);
        $finish();
    end


endmodule
