//`include "VX_define.vh"

module VX_ahb_tb; 

    parameter PERIOD = 10; 
    logic CLK = 0;
    logic nRST; // Active high  

    always #(PERIOD/2) CLK = ~CLK; 

    ahb_if ahbif(.HCLK(CLK), .HRESETn(nRST)); 

    VX_ahb DUT(CLK, ~nRST, ahbif); 

    // Assume same clock
    initial begin 
        nRST = 1'b0; 
        ahbif.HREADY = 1'b0; 
        ahbif.HRESP = 2'b0; 
        ahbif.HRDATA = 32'b0; 
        
        #(PERIOD); 
        ahbif.HREADY = 1'b1; 
        ahbif.HRESP = 2'b0; 
        ahbif.HRDATA = 32'hABCD1234; 
        #(PERIOD); 
        $$display("HELLO WORLD");
        
        $stop(); 
    end 

endmodule 