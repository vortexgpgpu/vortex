/* verilog_memcomp Version: c0.4.0-EAC */
/* common_memcomp Version: c0.1.0-EAC */
/* lang compiler Version: 4.1.6-EAC2 Oct 30 2012 16:32:37 */
//
//       CONFIDENTIAL AND PROPRIETARY SOFTWARE OF ARM PHYSICAL IP, INC.
//      
//       Copyright (c) 1993 - 2019 ARM Physical IP, Inc.  All Rights Reserved.
//      
//       Use of this Software is subject to the terms and conditions of the
//       applicable license agreement with ARM Physical IP, Inc.
//       In addition, this Software is protected by patents, copyright law 
//       and international treaties.
//      
//       The copyright notice(s) in this Software does not indicate actual or
//       intended publication of this Software.
//
//      Verilog model for High Density Two Port Register File SVT MVT Compiler
//
//       Instance Name:              rf2_256x19_wm0
//       Words:                      256
//       Bits:                       19
//       Mux:                        2
//       Drive:                      6
//       Write Mask:                 Off
//       Write Thru:                 Off
//       Extra Margin Adjustment:    On
//       Test Muxes                  On
//       Power Gating:               Off
//       Retention:                  On
//       Pipeline:                   Off
//       Read Disturb Test:	        Off
//       
//       Creation Date:  Sun Oct 20 14:44:21 2019
//       Version: 	r4p0
//
//      Modeling Assumptions: This model supports full gate level simulation
//          including proper x-handling and timing check behavior.  Unit
//          delay timing is included in the model. Back-annotation of SDF
//          (v3.0 or v2.1) is supported.  SDF can be created utilyzing the delay
//          calculation views provided with this generator and supported
//          delay calculators.  All buses are modeled [MSB:LSB].  All 
//          ports are padded with Verilog primitives.
//
//      Modeling Limitations: None.
//
//      Known Bugs: None.
//
//      Known Work Arounds: N/A
//

`define ARM_UD_MODEL
`timescale 1 ns/1 ps
`define ARM_MEM_PROP 1.000
`define ARM_MEM_RETAIN 1.000
`define ARM_MEM_PERIOD 3.000
`define ARM_MEM_WIDTH 1.000
`define ARM_MEM_SETUP 1.000
`define ARM_MEM_HOLD 0.500
`define ARM_MEM_COLLISION 3.000
// If ARM_HVM_MODEL is defined at Simulator Command Line, it Selects the Hierarchical Verilog Model
`ifdef ARM_HVM_MODEL


module datapath_latch_rf2_256x19_wm0 (CLK,Q_update,SE,SI,D,DFTRAMBYP,mem_path,XQ,Q);
	input CLK,Q_update,SE,SI,D,DFTRAMBYP,mem_path,XQ;
	output Q;

	reg    D_int;
	reg    Q;

   //  Model PHI2 portion
   always @(CLK or SE or SI or D) begin
      if (CLK === 1'b0) begin
         if (SE===1'b1)
           D_int=SI;
         else if (SE===1'bx)
           D_int=1'bx;
         else
           D_int=D;
      end
   end

   // model output side of RAM latch
   always @(posedge Q_update or posedge XQ) begin
      #0;
      if (XQ===1'b0) begin
         if (DFTRAMBYP===1'b1)
           Q=D_int;
         else
           Q=mem_path;
      end
      else
        Q=1'bx;
   end
endmodule // datapath_latch_rf2_256x19_wm0

// If ARM_UD_MODEL is defined at Simulator Command Line, it Selects the Fast Functional Model
`ifdef ARM_UD_MODEL

// Following parameter Values can be overridden at Simulator Command Line.

// ARM_UD_DP Defines the delay through Data Paths, for Memory Models it represents BIST MUX output delays.
`ifdef ARM_UD_DP
`else
`define ARM_UD_DP #0.001
`endif
// ARM_UD_CP Defines the delay through Clock Path Cells, for Memory Models it is not used.
`ifdef ARM_UD_CP
`else
`define ARM_UD_CP
`endif
// ARM_UD_SEQ Defines the delay through the Memory, for Memory Models it is used for CLK->Q delays.
`ifdef ARM_UD_SEQ
`else
`define ARM_UD_SEQ #0.01
`endif

`celldefine
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module rf2_256x19_wm0 (VDDCE, VDDPE, VSSE, CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA,
    CENA, AA, CLKB, CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB,
    TAB, TDB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`else
module rf2_256x19_wm0 (CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA, CENA, AA, CLKB,
    CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB, TAB, TDB, RET1N,
    SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 19;
  parameter WORDS = 256;
  parameter MUX = 2;
  parameter MEM_WIDTH = 38; // redun block size 2, 18 on left, 20 on right
  parameter MEM_HEIGHT = 128;
  parameter WP_SIZE = 19 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 0;
  parameter UPMS_WIDTH = 1;

  output  CENYA;
  output [7:0] AYA;
  output  CENYB;
  output [7:0] AYB;
  output [18:0] QA;
  output [1:0] SOA;
  output [1:0] SOB;
  input  CLKA;
  input  CENA;
  input [7:0] AA;
  input  CLKB;
  input  CENB;
  input [7:0] AB;
  input [18:0] DB;
  input [2:0] EMAA;
  input  EMASA;
  input [2:0] EMAB;
  input  TENA;
  input  TCENA;
  input [7:0] TAA;
  input  TENB;
  input  TCENB;
  input [7:0] TAB;
  input [18:0] TDB;
  input  RET1N;
  input [1:0] SIA;
  input  SEA;
  input  DFTRAMBYP;
  input [1:0] SIB;
  input  SEB;
  input  COLLDISN;
`ifdef POWER_PINS
  inout VDDCE;
  inout VDDPE;
  inout VSSE;
`endif

  reg pre_charge_st;
  reg pre_charge_st_a;
  reg pre_charge_st_b;
  integer row_address;
  integer mux_address;
  initial row_address = 0;
  initial mux_address = 0;
  reg [37:0] mem [0:127];
  reg [37:0] row, row_t;
  reg LAST_CLKA;
  reg [37:0] row_mask;
  reg [37:0] new_data;
  reg [37:0] data_out;
  reg [18:0] readLatch0;
  reg [18:0] shifted_readLatch0;
  reg  read_mux_sel0_p2;
  reg [18:0] readLatch1;
  reg [18:0] shifted_readLatch1;
  reg  read_mux_sel1_p2;
  reg LAST_CLKB;
  wire [18:0] QA_int;
  reg XQA, QA_update;
  reg [18:0] mem_path;
  reg XDB_sh, DB_sh_update;
  wire [18:0] DB_int_bmux;
  reg [18:0] writeEnable;
  real previous_CLKA;
  real previous_CLKB;
  initial previous_CLKA = 0;
  initial previous_CLKB = 0;
  reg READ_WRITE, WRITE_WRITE, READ_READ, ROW_CC, COL_CC;
  reg READ_WRITE_1, WRITE_WRITE_1, READ_READ_1;
  reg  cont_flag0_int;
  reg  cont_flag1_int;
  initial cont_flag0_int = 1'b0;
  initial cont_flag1_int = 1'b0;
  reg clk0_int;
  reg clk1_int;

  wire  CENYA_;
  wire [7:0] AYA_;
  wire  CENYB_;
  wire [7:0] AYB_;
  wire [18:0] QA_;
  wire [1:0] SOA_;
  wire [1:0] SOB_;
 wire  CLKA_;
  wire  CENA_;
  reg  CENA_int;
  reg  CENA_p2;
  wire [7:0] AA_;
  reg [7:0] AA_int;
 wire  CLKB_;
  wire  CENB_;
  reg  CENB_int;
  reg  CENB_p2;
  wire [7:0] AB_;
  reg [7:0] AB_int;
  wire [18:0] DB_;
  reg [18:0] DB_int;
  wire [18:0] DB_int_sh;
  reg [18:0] DB_int_sh_int;
  wire [2:0] EMAA_;
  reg [2:0] EMAA_int;
  wire  EMASA_;
  reg  EMASA_int;
  wire [2:0] EMAB_;
  reg [2:0] EMAB_int;
  wire  TENA_;
  reg  TENA_int;
  wire  TCENA_;
  reg  TCENA_int;
  reg  TCENA_p2;
  wire [7:0] TAA_;
  reg [7:0] TAA_int;
  wire  TENB_;
  reg  TENB_int;
  wire  TCENB_;
  reg  TCENB_int;
  reg  TCENB_p2;
  wire [7:0] TAB_;
  reg [7:0] TAB_int;
  wire [18:0] TDB_;
  reg [18:0] TDB_int;
  wire  RET1N_;
  reg  RET1N_int;
  wire [1:0] SIA_;
  wire [1:0] SIA_int;
  wire  SEA_;
  reg  SEA_int;
  wire  DFTRAMBYP_;
  reg  DFTRAMBYP_int;
  reg  DFTRAMBYP_p2;
  wire [1:0] SIB_;
  reg [1:0] SIB_int;
  wire  SEB_;
  reg  SEB_int;
  wire  COLLDISN_;
  reg  COLLDISN_int;

  assign CENYA = CENYA_; 
  assign AYA[0] = AYA_[0]; 
  assign AYA[1] = AYA_[1]; 
  assign AYA[2] = AYA_[2]; 
  assign AYA[3] = AYA_[3]; 
  assign AYA[4] = AYA_[4]; 
  assign AYA[5] = AYA_[5]; 
  assign AYA[6] = AYA_[6]; 
  assign AYA[7] = AYA_[7]; 
  assign CENYB = CENYB_; 
  assign AYB[0] = AYB_[0]; 
  assign AYB[1] = AYB_[1]; 
  assign AYB[2] = AYB_[2]; 
  assign AYB[3] = AYB_[3]; 
  assign AYB[4] = AYB_[4]; 
  assign AYB[5] = AYB_[5]; 
  assign AYB[6] = AYB_[6]; 
  assign AYB[7] = AYB_[7]; 
  assign QA[0] = QA_[0]; 
  assign QA[1] = QA_[1]; 
  assign QA[2] = QA_[2]; 
  assign QA[3] = QA_[3]; 
  assign QA[4] = QA_[4]; 
  assign QA[5] = QA_[5]; 
  assign QA[6] = QA_[6]; 
  assign QA[7] = QA_[7]; 
  assign QA[8] = QA_[8]; 
  assign QA[9] = QA_[9]; 
  assign QA[10] = QA_[10]; 
  assign QA[11] = QA_[11]; 
  assign QA[12] = QA_[12]; 
  assign QA[13] = QA_[13]; 
  assign QA[14] = QA_[14]; 
  assign QA[15] = QA_[15]; 
  assign QA[16] = QA_[16]; 
  assign QA[17] = QA_[17]; 
  assign QA[18] = QA_[18]; 
  assign SOA[0] = SOA_[0]; 
  assign SOA[1] = SOA_[1]; 
  assign SOB[0] = SOB_[0]; 
  assign SOB[1] = SOB_[1]; 
  assign CLKA_ = CLKA;
  assign CENA_ = CENA;
  assign AA_[0] = AA[0];
  assign AA_[1] = AA[1];
  assign AA_[2] = AA[2];
  assign AA_[3] = AA[3];
  assign AA_[4] = AA[4];
  assign AA_[5] = AA[5];
  assign AA_[6] = AA[6];
  assign AA_[7] = AA[7];
  assign CLKB_ = CLKB;
  assign CENB_ = CENB;
  assign AB_[0] = AB[0];
  assign AB_[1] = AB[1];
  assign AB_[2] = AB[2];
  assign AB_[3] = AB[3];
  assign AB_[4] = AB[4];
  assign AB_[5] = AB[5];
  assign AB_[6] = AB[6];
  assign AB_[7] = AB[7];
  assign DB_[0] = DB[0];
  assign DB_[1] = DB[1];
  assign DB_[2] = DB[2];
  assign DB_[3] = DB[3];
  assign DB_[4] = DB[4];
  assign DB_[5] = DB[5];
  assign DB_[6] = DB[6];
  assign DB_[7] = DB[7];
  assign DB_[8] = DB[8];
  assign DB_[9] = DB[9];
  assign DB_[10] = DB[10];
  assign DB_[11] = DB[11];
  assign DB_[12] = DB[12];
  assign DB_[13] = DB[13];
  assign DB_[14] = DB[14];
  assign DB_[15] = DB[15];
  assign DB_[16] = DB[16];
  assign DB_[17] = DB[17];
  assign DB_[18] = DB[18];
  assign EMAA_[0] = EMAA[0];
  assign EMAA_[1] = EMAA[1];
  assign EMAA_[2] = EMAA[2];
  assign EMASA_ = EMASA;
  assign EMAB_[0] = EMAB[0];
  assign EMAB_[1] = EMAB[1];
  assign EMAB_[2] = EMAB[2];
  assign TENA_ = TENA;
  assign TCENA_ = TCENA;
  assign TAA_[0] = TAA[0];
  assign TAA_[1] = TAA[1];
  assign TAA_[2] = TAA[2];
  assign TAA_[3] = TAA[3];
  assign TAA_[4] = TAA[4];
  assign TAA_[5] = TAA[5];
  assign TAA_[6] = TAA[6];
  assign TAA_[7] = TAA[7];
  assign TENB_ = TENB;
  assign TCENB_ = TCENB;
  assign TAB_[0] = TAB[0];
  assign TAB_[1] = TAB[1];
  assign TAB_[2] = TAB[2];
  assign TAB_[3] = TAB[3];
  assign TAB_[4] = TAB[4];
  assign TAB_[5] = TAB[5];
  assign TAB_[6] = TAB[6];
  assign TAB_[7] = TAB[7];
  assign TDB_[0] = TDB[0];
  assign TDB_[1] = TDB[1];
  assign TDB_[2] = TDB[2];
  assign TDB_[3] = TDB[3];
  assign TDB_[4] = TDB[4];
  assign TDB_[5] = TDB[5];
  assign TDB_[6] = TDB[6];
  assign TDB_[7] = TDB[7];
  assign TDB_[8] = TDB[8];
  assign TDB_[9] = TDB[9];
  assign TDB_[10] = TDB[10];
  assign TDB_[11] = TDB[11];
  assign TDB_[12] = TDB[12];
  assign TDB_[13] = TDB[13];
  assign TDB_[14] = TDB[14];
  assign TDB_[15] = TDB[15];
  assign TDB_[16] = TDB[16];
  assign TDB_[17] = TDB[17];
  assign TDB_[18] = TDB[18];
  assign RET1N_ = RET1N;
  assign SIA_[0] = SIA[0];
  assign SIA_[1] = SIA[1];
  assign SEA_ = SEA;
  assign DFTRAMBYP_ = DFTRAMBYP;
  assign SIB_[0] = SIB[0];
  assign SIB_[1] = SIB[1];
  assign SEB_ = SEB;
  assign COLLDISN_ = COLLDISN;

  assign `ARM_UD_DP CENYA_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENA_ ? CENA_ : TCENA_)) : 1'bx;
  assign `ARM_UD_DP AYA_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENA_ ? AA_ : TAA_)) : {8{1'bx}};
  assign `ARM_UD_DP CENYB_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENB_ ? CENB_ : TCENB_)) : 1'bx;
  assign `ARM_UD_DP AYB_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENB_ ? AB_ : TAB_)) : {8{1'bx}};
  assign `ARM_UD_SEQ QA_ = (RET1N_ | pre_charge_st) ? ((QA_int)) : {19{1'bx}};
  assign `ARM_UD_DP SOA_ = (RET1N_ | pre_charge_st) ? ({QA_[18], QA_[0]}) : {2{1'bx}};
  assign `ARM_UD_DP SOB_ = (RET1N_ | pre_charge_st) ? ({DB_int_sh[18], DB_int_sh[0]}) : {2{1'bx}};

// If INITIALIZE_MEMORY is defined at Simulator Command Line, it Initializes the Memory with all ZEROS.
`ifdef INITIALIZE_MEMORY
  integer i;
  initial begin
    #0;
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'b0}};
  end
`endif
  always @ (EMAA_) begin
  	if(EMAA_ < 3) 
   	$display("Warning: Set Value for EMAA doesn't match Default value 3 in %m at %0t", $time);
  end
  always @ (EMASA_) begin
  	if(EMASA_ < 0) 
   	$display("Warning: Set Value for EMASA doesn't match Default value 0 in %m at %0t", $time);
  end
  always @ (EMAB_) begin
  	if(EMAB_ < 3) 
   	$display("Warning: Set Value for EMAB doesn't match Default value 3 in %m at %0t", $time);
  end

  task failedWrite;
  input port_f;
  integer i;
  begin
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'bx}};
  end
  endtask

  function isBitX;
    input bitval;
    begin
      isBitX = ( bitval===1'bx || bitval===1'bz ) ? 1'b1 : 1'b0;
    end
  endfunction

  function isBit1;
    input bitval;
    begin
      isBit1 = ( bitval===1'b1 ) ? 1'b1 : 1'b0;
    end
  endfunction


task loadmem;
	input [1000*8-1:0] filename;
	reg [BITS-1:0] memld [0:WORDS-1];
	integer i;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	$readmemb(filename, memld);
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  wordtemp = memld[i];
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  	end
  end
  end
  endtask

task dumpmem;
	input [1000*8-1:0] filename_dump;
	integer i, dump_file_desc;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	dump_file_desc = $fopen(filename_dump, "w");
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      mem_path = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
        	XQA = 1'b0; QA_update = 1'b1;
   	$fdisplay(dump_file_desc, "%b", QA_int);
  end
  	end
    $fclose(dump_file_desc);
  end
  endtask

task loadaddr;
	input [7:0] load_addr;
	input [18:0] load_data;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  wordtemp = load_data;
	  Atemp = load_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  end
  end
  endtask

task dumpaddr;
	output [18:0] dump_data;
	input [7:0] dump_addr;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  Atemp = dump_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      mem_path = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
        	XQA = 1'b0; QA_update = 1'b1;
   	dump_data = QA_int;
  	end
  end
  endtask


  task ReadA;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'bx) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'b1) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0 && (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAA_int & isBit1(DFTRAMBYP_int)), (EMASA_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if ((AA_int >= WORDS) && (CENA_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
        XQA = 0 ? 1'b0 : 1'b1; QA_update = 0 ? 1'b0 : 1'b1;
    end else if (CENA_int === 1'b0 && (^AA_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if (DFTRAMBYP_int !== 1'b1) begin
      mux_address = (AA_int & 1'b1);
      row_address = (AA_int >> 1);
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      data_out = (row >> mux_address);
      mem_path = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
        	XQA = 1'b0; QA_update = 1'b1;
      end
        if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'b0) begin
        end else if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'bx) begin
        	XQA = 1'b1; QA_update = 1'b1;
        end
      if( isBitX(DFTRAMBYP_int) ) begin
        XQA = 1'b1; QA_update = 1'b1;
      end
      if( isBitX(SEA_int) && DFTRAMBYP_int === 1'b1 ) begin
        XQA = 1'b1; QA_update = 1'b1;
      end
      if(isBitX(DFTRAMBYP_int)) begin
        XQA = 1'b1; QA_update = 1'b1;
        failedWrite(0);
      end
    end
  end
  endtask

  task WriteB;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'bx) begin
      failedWrite(1);
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'b1) begin
      failedWrite(1);
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0 && (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAB_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) begin
      failedWrite(1);
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if ((AB_int >= WORDS) && (CENB_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
    end else if (CENB_int === 1'b0 && (^AB_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(1);
    end else if (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int))
        DB_int = {19{1'bx}};

      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int)) begin
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
      end
      mux_address = (AB_int & 1'b1);
      row_address = (AB_int >> 1);
      if (DFTRAMBYP_int !== 1'b1) begin
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      end
      if(isBitX(DFTRAMBYP_int)) begin
        writeEnable = {19{1'bx}};
        DB_int = {19{1'bx}};
      end else
          writeEnable = ~ {19{CENB_int}};
      row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
        1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
        1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
        1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
        1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
      new_data =  ( {1'b0, DB_int[18], 1'b0, DB_int[17], 1'b0, DB_int[16], 1'b0, DB_int[15],
        1'b0, DB_int[14], 1'b0, DB_int[13], 1'b0, DB_int[12], 1'b0, DB_int[11], 1'b0, DB_int[10],
        1'b0, DB_int[9], 1'b0, DB_int[8], 1'b0, DB_int[7], 1'b0, DB_int[6], 1'b0, DB_int[5],
        1'b0, DB_int[4], 1'b0, DB_int[3], 1'b0, DB_int[2], 1'b0, DB_int[1], 1'b0, DB_int[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        if (DFTRAMBYP_int === 1'b1 && (SEB_int === 1'b0 || SEB_int === 1'bx)) begin
        end else begin
        	mem[row_address] = row;
        end
    end
  end
  endtask
  always @ (CENA_ or TCENA_ or TENA_ or DFTRAMBYP_ or CLKA_) begin
  	if(CLKA_ == 1'b0) begin
  		CENA_p2 = CENA_;
  		TCENA_p2 = TCENA_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (VDDCE) begin
      if (VDDCE != 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDCE should be powered down after VDDPE, Illegal power down sequencing in %m at %0t", $time);
       end
        $display("In PowerDown Mode in %m at %0t", $time);
        failedWrite(0);
      end
      if (VDDCE == 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDPE should be powered up after VDDCE in %m at %0t", $time);
        $display("Illegal power up sequencing in %m at %0t", $time);
       end
        failedWrite(0);
      end
  end
`endif
`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_a == 1'b1 && (CENA_ === 1'bx || TCENA_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKA_ === 1'bx)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_a = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(0);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
        XQA = 1'b1; QA_update = 1'b1;
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_a == 1'b1) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
        XQA = 1'b1; QA_update = 1'b1;
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
    #0;
        QA_update = 1'b0;
  end

  always @ (CLKB_ or DFTRAMBYP_p2) begin
  	#0;
  	if(CLKB_ == 1'b1 && (DFTRAMBYP_int === 1'b1 || CENB_int != 1'b1)) begin
  	  if (RET1N_ == 1'b1) begin
	        DB_sh_update = 1'b1; 
  	  end
  	end
  end

  always @ CLKA_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKA_ === 1'bx || CLKA_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if ((CLKA_ === 1'b1 || CLKA_ === 1'b0) && LAST_CLKA === 1'bx) begin
      XQA = 1'b0; QA_update = 1'b0; 
    end else if (CLKA_ === 1'b1 && LAST_CLKA === 1'b0) begin
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
  end else begin
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
  if (RET1N_ == 1'b1) begin
        XQA = 1'b0; QA_update = 1'b1;
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b0) begin
  if (RET1N_ == 1'b1) begin
        XQA = 1'b0; QA_update = 1'b1;
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else begin
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
    ReadA;
      if (CENA_int === 1'b0) previous_CLKA = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int, 1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
      end
  end
    end else if (CLKA_ === 1'b0 && LAST_CLKA === 1'b1) begin
      QA_update = 1'b0;
      XQA = 1'b0;
    end
  end
    LAST_CLKA = CLKA_;
  end

  assign SIA_int = SEA_ ? SIA_ : {2{1'b0}};

  datapath_latch_rf2_256x19_wm0 uDQA0 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[1]), .D(QA_int[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[0]), .XQ(XQA), .Q(QA_int[0]));
  datapath_latch_rf2_256x19_wm0 uDQA1 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[2]), .D(QA_int[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[1]), .XQ(XQA), .Q(QA_int[1]));
  datapath_latch_rf2_256x19_wm0 uDQA2 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[3]), .D(QA_int[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[2]), .XQ(XQA), .Q(QA_int[2]));
  datapath_latch_rf2_256x19_wm0 uDQA3 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[4]), .D(QA_int[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[3]), .XQ(XQA), .Q(QA_int[3]));
  datapath_latch_rf2_256x19_wm0 uDQA4 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[5]), .D(QA_int[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[4]), .XQ(XQA), .Q(QA_int[4]));
  datapath_latch_rf2_256x19_wm0 uDQA5 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[6]), .D(QA_int[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[5]), .XQ(XQA), .Q(QA_int[5]));
  datapath_latch_rf2_256x19_wm0 uDQA6 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[7]), .D(QA_int[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[6]), .XQ(XQA), .Q(QA_int[6]));
  datapath_latch_rf2_256x19_wm0 uDQA7 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[8]), .D(QA_int[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[7]), .XQ(XQA), .Q(QA_int[7]));
  datapath_latch_rf2_256x19_wm0 uDQA8 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(SIA_int[0]), .D(1'b0), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[8]), .XQ(XQA), .Q(QA_int[8]));
  datapath_latch_rf2_256x19_wm0 uDQA9 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(SIA_int[1]), .D(1'b0), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[9]), .XQ(XQA), .Q(QA_int[9]));
  datapath_latch_rf2_256x19_wm0 uDQA10 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[9]), .D(QA_int[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[10]), .XQ(XQA), .Q(QA_int[10]));
  datapath_latch_rf2_256x19_wm0 uDQA11 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[10]), .D(QA_int[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[11]), .XQ(XQA), .Q(QA_int[11]));
  datapath_latch_rf2_256x19_wm0 uDQA12 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[11]), .D(QA_int[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[12]), .XQ(XQA), .Q(QA_int[12]));
  datapath_latch_rf2_256x19_wm0 uDQA13 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[12]), .D(QA_int[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[13]), .XQ(XQA), .Q(QA_int[13]));
  datapath_latch_rf2_256x19_wm0 uDQA14 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[13]), .D(QA_int[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[14]), .XQ(XQA), .Q(QA_int[14]));
  datapath_latch_rf2_256x19_wm0 uDQA15 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[14]), .D(QA_int[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[15]), .XQ(XQA), .Q(QA_int[15]));
  datapath_latch_rf2_256x19_wm0 uDQA16 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[15]), .D(QA_int[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[16]), .XQ(XQA), .Q(QA_int[16]));
  datapath_latch_rf2_256x19_wm0 uDQA17 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[16]), .D(QA_int[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[17]), .XQ(XQA), .Q(QA_int[17]));
  datapath_latch_rf2_256x19_wm0 uDQA18 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[17]), .D(QA_int[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[18]), .XQ(XQA), .Q(QA_int[18]));



  always @ (CENB_ or TCENB_ or TENB_ or DFTRAMBYP_ or CLKB_) begin
  	if(CLKB_ == 1'b0) begin
  		CENB_p2 = CENB_;
  		TCENB_p2 = TCENB_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_b == 1'b1 && (CENB_ === 1'bx || TCENB_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKB_ === 1'bx)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_b = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(1);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_b == 1'b1) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
    #0;
        QA_update = 1'b0;
        DB_sh_update = 1'b0; 
  end

  always @ CLKB_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKB_ === 1'bx || CLKB_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
    end else if ((CLKB_ === 1'b1 || CLKB_ === 1'b0) && LAST_CLKB === 1'bx) begin
       DB_sh_update = 1'b0;  XDB_sh = 1'b0;
    end else if (CLKB_ === 1'b1 && LAST_CLKB === 1'b0) begin
  if (RET1N_ == 1'b0) begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SEB_int = SEB_;
  end else begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SEB_int = SEB_;
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        XDB_sh = 1'b0; 
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
        XDB_sh = 1'b0; 
      end else begin
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        XDB_sh = 1'b0; 
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b0) begin
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
      end else begin
      WriteB;
      end
      if (CENB_int === 1'b0) previous_CLKB = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
    end
      end
    end else if (CLKB_ === 1'b0 && LAST_CLKB === 1'b1) begin
       DB_sh_update = 1'b0;  XDB_sh = 1'b0;
  end
  end
    LAST_CLKB = CLKB_;
  end

  assign DB_int_bmux = TENB_ ? DB_ : TDB_;

  datapath_latch_rf2_256x19_wm0 uDQB0 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[1]), .D(DB_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[0]), .XQ(XDB_sh), .Q(DB_int_sh[0]));
  datapath_latch_rf2_256x19_wm0 uDQB1 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[2]), .D(DB_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[1]), .XQ(XDB_sh), .Q(DB_int_sh[1]));
  datapath_latch_rf2_256x19_wm0 uDQB2 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[3]), .D(DB_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[2]), .XQ(XDB_sh), .Q(DB_int_sh[2]));
  datapath_latch_rf2_256x19_wm0 uDQB3 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[4]), .D(DB_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[3]), .XQ(XDB_sh), .Q(DB_int_sh[3]));
  datapath_latch_rf2_256x19_wm0 uDQB4 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[5]), .D(DB_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[4]), .XQ(XDB_sh), .Q(DB_int_sh[4]));
  datapath_latch_rf2_256x19_wm0 uDQB5 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[6]), .D(DB_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[5]), .XQ(XDB_sh), .Q(DB_int_sh[5]));
  datapath_latch_rf2_256x19_wm0 uDQB6 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[7]), .D(DB_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[6]), .XQ(XDB_sh), .Q(DB_int_sh[6]));
  datapath_latch_rf2_256x19_wm0 uDQB7 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[8]), .D(DB_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[7]), .XQ(XDB_sh), .Q(DB_int_sh[7]));
  datapath_latch_rf2_256x19_wm0 uDQB8 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(SIB_[0]), .D(DB_int_bmux[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[8]), .XQ(XDB_sh), .Q(DB_int_sh[8]));
  datapath_latch_rf2_256x19_wm0 uDQB9 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(SIB_[1]), .D(DB_int_bmux[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[9]), .XQ(XDB_sh), .Q(DB_int_sh[9]));
  datapath_latch_rf2_256x19_wm0 uDQB10 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[9]), .D(DB_int_bmux[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[10]), .XQ(XDB_sh), .Q(DB_int_sh[10]));
  datapath_latch_rf2_256x19_wm0 uDQB11 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[10]), .D(DB_int_bmux[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[11]), .XQ(XDB_sh), .Q(DB_int_sh[11]));
  datapath_latch_rf2_256x19_wm0 uDQB12 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[11]), .D(DB_int_bmux[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[12]), .XQ(XDB_sh), .Q(DB_int_sh[12]));
  datapath_latch_rf2_256x19_wm0 uDQB13 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[12]), .D(DB_int_bmux[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[13]), .XQ(XDB_sh), .Q(DB_int_sh[13]));
  datapath_latch_rf2_256x19_wm0 uDQB14 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[13]), .D(DB_int_bmux[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[14]), .XQ(XDB_sh), .Q(DB_int_sh[14]));
  datapath_latch_rf2_256x19_wm0 uDQB15 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[14]), .D(DB_int_bmux[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[15]), .XQ(XDB_sh), .Q(DB_int_sh[15]));
  datapath_latch_rf2_256x19_wm0 uDQB16 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[15]), .D(DB_int_bmux[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[16]), .XQ(XDB_sh), .Q(DB_int_sh[16]));
  datapath_latch_rf2_256x19_wm0 uDQB17 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[16]), .D(DB_int_bmux[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[17]), .XQ(XDB_sh), .Q(DB_int_sh[17]));
  datapath_latch_rf2_256x19_wm0 uDQB18 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[17]), .D(DB_int_bmux[18]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[18]), .XQ(XDB_sh), .Q(DB_int_sh[18]));



// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
 always @ (VDDCE or VDDPE or VSSE) begin
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
 end
`endif

  function row_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
    reg sameRow;
    reg sameMux;
    reg anyWrite;
  begin
    anyWrite = ((& wena) === 1'b1 && (& wenb) === 1'b1) ? 1'b0 : 1'b1;
    sameMux = (aa[0:0] == ab[0:0]) ? 1'b1 : 1'b0;
    if (aa[7:1] == ab[7:1]) begin
      sameRow = 1'b1;
    end else begin
      sameRow = 1'b0;
    end
    if (sameRow == 1'b1 && anyWrite == 1'b1)
      row_contention = 1'b1;
    else if (sameRow == 1'b1 && sameMux == 1'b1)
      row_contention = 1'b1;
    else
      row_contention = 1'b0;
  end
  endfunction

  function col_contention;
    input [7:0] aa;
    input [7:0] ab;
  begin
    if (aa[0:0] == ab[0:0])
      col_contention = 1'b1;
    else
      col_contention = 1'b0;
  end
  endfunction

  function is_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
  begin
    if ((& wena) === 1'b1 && (& wenb) === 1'b1) begin
      result = 1'b0;
    end else if (aa == ab) begin
      result = 1'b1;
    end else begin
      result = 1'b0;
    end
    is_contention = result;
  end
  endfunction


endmodule
`endcelldefine
`else
`celldefine
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module rf2_256x19_wm0 (VDDCE, VDDPE, VSSE, CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA,
    CENA, AA, CLKB, CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB,
    TAB, TDB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`else
module rf2_256x19_wm0 (CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA, CENA, AA, CLKB,
    CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB, TAB, TDB, RET1N,
    SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 19;
  parameter WORDS = 256;
  parameter MUX = 2;
  parameter MEM_WIDTH = 38; // redun block size 2, 18 on left, 20 on right
  parameter MEM_HEIGHT = 128;
  parameter WP_SIZE = 19 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 0;
  parameter UPMS_WIDTH = 1;

  output  CENYA;
  output [7:0] AYA;
  output  CENYB;
  output [7:0] AYB;
  output [18:0] QA;
  output [1:0] SOA;
  output [1:0] SOB;
  input  CLKA;
  input  CENA;
  input [7:0] AA;
  input  CLKB;
  input  CENB;
  input [7:0] AB;
  input [18:0] DB;
  input [2:0] EMAA;
  input  EMASA;
  input [2:0] EMAB;
  input  TENA;
  input  TCENA;
  input [7:0] TAA;
  input  TENB;
  input  TCENB;
  input [7:0] TAB;
  input [18:0] TDB;
  input  RET1N;
  input [1:0] SIA;
  input  SEA;
  input  DFTRAMBYP;
  input [1:0] SIB;
  input  SEB;
  input  COLLDISN;
`ifdef POWER_PINS
  inout VDDCE;
  inout VDDPE;
  inout VSSE;
`endif

  reg pre_charge_st;
  reg pre_charge_st_a;
  reg pre_charge_st_b;
  integer row_address;
  integer mux_address;
  initial row_address = 0;
  initial mux_address = 0;
  reg [37:0] mem [0:127];
  reg [37:0] row, row_t;
  reg LAST_CLKA;
  reg [37:0] row_mask;
  reg [37:0] new_data;
  reg [37:0] data_out;
  reg [18:0] readLatch0;
  reg [18:0] shifted_readLatch0;
  reg  read_mux_sel0_p2;
  reg [18:0] readLatch1;
  reg [18:0] shifted_readLatch1;
  reg  read_mux_sel1_p2;
  reg LAST_CLKB;
  wire [18:0] QA_int;
  reg XQA, QA_update;
  reg [18:0] mem_path;
  reg XDB_sh, DB_sh_update;
  wire [18:0] DB_int_bmux;
  reg [18:0] writeEnable;
  real previous_CLKA;
  real previous_CLKB;
  initial previous_CLKA = 0;
  initial previous_CLKB = 0;
  reg READ_WRITE, WRITE_WRITE, READ_READ, ROW_CC, COL_CC;
  reg READ_WRITE_1, WRITE_WRITE_1, READ_READ_1;
  reg  cont_flag0_int;
  reg  cont_flag1_int;
  initial cont_flag0_int = 1'b0;
  initial cont_flag1_int = 1'b0;

  reg NOT_CENA, NOT_AA7, NOT_AA6, NOT_AA5, NOT_AA4, NOT_AA3, NOT_AA2, NOT_AA1, NOT_AA0;
  reg NOT_CENB, NOT_AB7, NOT_AB6, NOT_AB5, NOT_AB4, NOT_AB3, NOT_AB2, NOT_AB1, NOT_AB0;
  reg NOT_DB18, NOT_DB17, NOT_DB16, NOT_DB15, NOT_DB14, NOT_DB13, NOT_DB12, NOT_DB11;
  reg NOT_DB10, NOT_DB9, NOT_DB8, NOT_DB7, NOT_DB6, NOT_DB5, NOT_DB4, NOT_DB3, NOT_DB2;
  reg NOT_DB1, NOT_DB0, NOT_EMAA2, NOT_EMAA1, NOT_EMAA0, NOT_EMASA, NOT_EMAB2, NOT_EMAB1;
  reg NOT_EMAB0, NOT_TENA, NOT_TCENA, NOT_TAA7, NOT_TAA6, NOT_TAA5, NOT_TAA4, NOT_TAA3;
  reg NOT_TAA2, NOT_TAA1, NOT_TAA0, NOT_TENB, NOT_TCENB, NOT_TAB7, NOT_TAB6, NOT_TAB5;
  reg NOT_TAB4, NOT_TAB3, NOT_TAB2, NOT_TAB1, NOT_TAB0, NOT_TDB18, NOT_TDB17, NOT_TDB16;
  reg NOT_TDB15, NOT_TDB14, NOT_TDB13, NOT_TDB12, NOT_TDB11, NOT_TDB10, NOT_TDB9, NOT_TDB8;
  reg NOT_TDB7, NOT_TDB6, NOT_TDB5, NOT_TDB4, NOT_TDB3, NOT_TDB2, NOT_TDB1, NOT_TDB0;
  reg NOT_SIA1, NOT_SIA0, NOT_SEA, NOT_DFTRAMBYP_CLKA, NOT_DFTRAMBYP_CLKB, NOT_RET1N;
  reg NOT_SIB1, NOT_SIB0, NOT_SEB, NOT_COLLDISN;
  reg NOT_CONTA, NOT_CLKA_PER, NOT_CLKA_MINH, NOT_CLKA_MINL, NOT_CONTB, NOT_CLKB_PER;
  reg NOT_CLKB_MINH, NOT_CLKB_MINL;
  reg clk0_int;
  reg clk1_int;

  wire  CENYA_;
  wire [7:0] AYA_;
  wire  CENYB_;
  wire [7:0] AYB_;
  wire [18:0] QA_;
  wire [1:0] SOA_;
  wire [1:0] SOB_;
 wire  CLKA_;
  wire  CENA_;
  reg  CENA_int;
  reg  CENA_p2;
  wire [7:0] AA_;
  reg [7:0] AA_int;
 wire  CLKB_;
  wire  CENB_;
  reg  CENB_int;
  reg  CENB_p2;
  wire [7:0] AB_;
  reg [7:0] AB_int;
  wire [18:0] DB_;
  reg [18:0] DB_int;
  wire [18:0] DB_int_sh;
  reg [18:0] DB_int_sh_int;
  wire [2:0] EMAA_;
  reg [2:0] EMAA_int;
  wire  EMASA_;
  reg  EMASA_int;
  wire [2:0] EMAB_;
  reg [2:0] EMAB_int;
  wire  TENA_;
  reg  TENA_int;
  wire  TCENA_;
  reg  TCENA_int;
  reg  TCENA_p2;
  wire [7:0] TAA_;
  reg [7:0] TAA_int;
  wire  TENB_;
  reg  TENB_int;
  wire  TCENB_;
  reg  TCENB_int;
  reg  TCENB_p2;
  wire [7:0] TAB_;
  reg [7:0] TAB_int;
  wire [18:0] TDB_;
  reg [18:0] TDB_int;
  wire  RET1N_;
  reg  RET1N_int;
  wire [1:0] SIA_;
  wire [1:0] SIA_int;
  wire  SEA_;
  reg  SEA_int;
  wire  DFTRAMBYP_;
  reg  DFTRAMBYP_int;
  reg  DFTRAMBYP_p2;
  wire [1:0] SIB_;
  reg [1:0] SIB_int;
  wire  SEB_;
  reg  SEB_int;
  wire  COLLDISN_;
  reg  COLLDISN_int;

  buf B0(CENYA, CENYA_);
  buf B1(AYA[0], AYA_[0]);
  buf B2(AYA[1], AYA_[1]);
  buf B3(AYA[2], AYA_[2]);
  buf B4(AYA[3], AYA_[3]);
  buf B5(AYA[4], AYA_[4]);
  buf B6(AYA[5], AYA_[5]);
  buf B7(AYA[6], AYA_[6]);
  buf B8(AYA[7], AYA_[7]);
  buf B9(CENYB, CENYB_);
  buf B10(AYB[0], AYB_[0]);
  buf B11(AYB[1], AYB_[1]);
  buf B12(AYB[2], AYB_[2]);
  buf B13(AYB[3], AYB_[3]);
  buf B14(AYB[4], AYB_[4]);
  buf B15(AYB[5], AYB_[5]);
  buf B16(AYB[6], AYB_[6]);
  buf B17(AYB[7], AYB_[7]);
  buf B18(QA[0], QA_[0]);
  buf B19(QA[1], QA_[1]);
  buf B20(QA[2], QA_[2]);
  buf B21(QA[3], QA_[3]);
  buf B22(QA[4], QA_[4]);
  buf B23(QA[5], QA_[5]);
  buf B24(QA[6], QA_[6]);
  buf B25(QA[7], QA_[7]);
  buf B26(QA[8], QA_[8]);
  buf B27(QA[9], QA_[9]);
  buf B28(QA[10], QA_[10]);
  buf B29(QA[11], QA_[11]);
  buf B30(QA[12], QA_[12]);
  buf B31(QA[13], QA_[13]);
  buf B32(QA[14], QA_[14]);
  buf B33(QA[15], QA_[15]);
  buf B34(QA[16], QA_[16]);
  buf B35(QA[17], QA_[17]);
  buf B36(QA[18], QA_[18]);
  buf B37(SOA[0], SOA_[0]);
  buf B38(SOA[1], SOA_[1]);
  buf B39(SOB[0], SOB_[0]);
  buf B40(SOB[1], SOB_[1]);
  buf B41(CLKA_, CLKA);
  buf B42(CENA_, CENA);
  buf B43(AA_[0], AA[0]);
  buf B44(AA_[1], AA[1]);
  buf B45(AA_[2], AA[2]);
  buf B46(AA_[3], AA[3]);
  buf B47(AA_[4], AA[4]);
  buf B48(AA_[5], AA[5]);
  buf B49(AA_[6], AA[6]);
  buf B50(AA_[7], AA[7]);
  buf B51(CLKB_, CLKB);
  buf B52(CENB_, CENB);
  buf B53(AB_[0], AB[0]);
  buf B54(AB_[1], AB[1]);
  buf B55(AB_[2], AB[2]);
  buf B56(AB_[3], AB[3]);
  buf B57(AB_[4], AB[4]);
  buf B58(AB_[5], AB[5]);
  buf B59(AB_[6], AB[6]);
  buf B60(AB_[7], AB[7]);
  buf B61(DB_[0], DB[0]);
  buf B62(DB_[1], DB[1]);
  buf B63(DB_[2], DB[2]);
  buf B64(DB_[3], DB[3]);
  buf B65(DB_[4], DB[4]);
  buf B66(DB_[5], DB[5]);
  buf B67(DB_[6], DB[6]);
  buf B68(DB_[7], DB[7]);
  buf B69(DB_[8], DB[8]);
  buf B70(DB_[9], DB[9]);
  buf B71(DB_[10], DB[10]);
  buf B72(DB_[11], DB[11]);
  buf B73(DB_[12], DB[12]);
  buf B74(DB_[13], DB[13]);
  buf B75(DB_[14], DB[14]);
  buf B76(DB_[15], DB[15]);
  buf B77(DB_[16], DB[16]);
  buf B78(DB_[17], DB[17]);
  buf B79(DB_[18], DB[18]);
  buf B80(EMAA_[0], EMAA[0]);
  buf B81(EMAA_[1], EMAA[1]);
  buf B82(EMAA_[2], EMAA[2]);
  buf B83(EMASA_, EMASA);
  buf B84(EMAB_[0], EMAB[0]);
  buf B85(EMAB_[1], EMAB[1]);
  buf B86(EMAB_[2], EMAB[2]);
  buf B87(TENA_, TENA);
  buf B88(TCENA_, TCENA);
  buf B89(TAA_[0], TAA[0]);
  buf B90(TAA_[1], TAA[1]);
  buf B91(TAA_[2], TAA[2]);
  buf B92(TAA_[3], TAA[3]);
  buf B93(TAA_[4], TAA[4]);
  buf B94(TAA_[5], TAA[5]);
  buf B95(TAA_[6], TAA[6]);
  buf B96(TAA_[7], TAA[7]);
  buf B97(TENB_, TENB);
  buf B98(TCENB_, TCENB);
  buf B99(TAB_[0], TAB[0]);
  buf B100(TAB_[1], TAB[1]);
  buf B101(TAB_[2], TAB[2]);
  buf B102(TAB_[3], TAB[3]);
  buf B103(TAB_[4], TAB[4]);
  buf B104(TAB_[5], TAB[5]);
  buf B105(TAB_[6], TAB[6]);
  buf B106(TAB_[7], TAB[7]);
  buf B107(TDB_[0], TDB[0]);
  buf B108(TDB_[1], TDB[1]);
  buf B109(TDB_[2], TDB[2]);
  buf B110(TDB_[3], TDB[3]);
  buf B111(TDB_[4], TDB[4]);
  buf B112(TDB_[5], TDB[5]);
  buf B113(TDB_[6], TDB[6]);
  buf B114(TDB_[7], TDB[7]);
  buf B115(TDB_[8], TDB[8]);
  buf B116(TDB_[9], TDB[9]);
  buf B117(TDB_[10], TDB[10]);
  buf B118(TDB_[11], TDB[11]);
  buf B119(TDB_[12], TDB[12]);
  buf B120(TDB_[13], TDB[13]);
  buf B121(TDB_[14], TDB[14]);
  buf B122(TDB_[15], TDB[15]);
  buf B123(TDB_[16], TDB[16]);
  buf B124(TDB_[17], TDB[17]);
  buf B125(TDB_[18], TDB[18]);
  buf B126(RET1N_, RET1N);
  buf B127(SIA_[0], SIA[0]);
  buf B128(SIA_[1], SIA[1]);
  buf B129(SEA_, SEA);
  buf B130(DFTRAMBYP_, DFTRAMBYP);
  buf B131(SIB_[0], SIB[0]);
  buf B132(SIB_[1], SIB[1]);
  buf B133(SEB_, SEB);
  buf B134(COLLDISN_, COLLDISN);

  assign CENYA_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENA_ ? CENA_ : TCENA_)) : 1'bx;
  assign AYA_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENA_ ? AA_ : TAA_)) : {8{1'bx}};
  assign CENYB_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENB_ ? CENB_ : TCENB_)) : 1'bx;
  assign AYB_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENB_ ? AB_ : TAB_)) : {8{1'bx}};
  assign QA_ = (RET1N_ | pre_charge_st) ? ((QA_int)) : {19{1'bx}};
  assign SOA_ = (RET1N_ | pre_charge_st) ? ({QA_[18], QA_[0]}) : {2{1'bx}};
  assign SOB_ = (RET1N_ | pre_charge_st) ? ({DB_int_sh[18], DB_int_sh[0]}) : {2{1'bx}};

// If INITIALIZE_MEMORY is defined at Simulator Command Line, it Initializes the Memory with all ZEROS.
`ifdef INITIALIZE_MEMORY
  integer i;
  initial begin
    #0;
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'b0}};
  end
`endif
  always @ (EMAA_) begin
  	if(EMAA_ < 3) 
   	$display("Warning: Set Value for EMAA doesn't match Default value 3 in %m at %0t", $time);
  end
  always @ (EMASA_) begin
  	if(EMASA_ < 0) 
   	$display("Warning: Set Value for EMASA doesn't match Default value 0 in %m at %0t", $time);
  end
  always @ (EMAB_) begin
  	if(EMAB_ < 3) 
   	$display("Warning: Set Value for EMAB doesn't match Default value 3 in %m at %0t", $time);
  end

  task failedWrite;
  input port_f;
  integer i;
  begin
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'bx}};
  end
  endtask

  function isBitX;
    input bitval;
    begin
      isBitX = ( bitval===1'bx || bitval===1'bz ) ? 1'b1 : 1'b0;
    end
  endfunction

  function isBit1;
    input bitval;
    begin
      isBit1 = ( bitval===1'b1 ) ? 1'b1 : 1'b0;
    end
  endfunction


task loadmem;
	input [1000*8-1:0] filename;
	reg [BITS-1:0] memld [0:WORDS-1];
	integer i;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	$readmemb(filename, memld);
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  wordtemp = memld[i];
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  	end
  end
  end
  endtask

task dumpmem;
	input [1000*8-1:0] filename_dump;
	integer i, dump_file_desc;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	dump_file_desc = $fopen(filename_dump, "w");
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      mem_path = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
        	XQA = 1'b0; QA_update = 1'b1;
   	$fdisplay(dump_file_desc, "%b", QA_int);
  end
  	end
    $fclose(dump_file_desc);
  end
  endtask

task loadaddr;
	input [7:0] load_addr;
	input [18:0] load_data;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  wordtemp = load_data;
	  Atemp = load_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  end
  end
  endtask

task dumpaddr;
	output [18:0] dump_data;
	input [7:0] dump_addr;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  Atemp = dump_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      mem_path = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
        	XQA = 1'b0; QA_update = 1'b1;
   	dump_data = QA_int;
  	end
  end
  endtask


  task ReadA;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'bx) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'b1) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0 && (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAA_int & isBit1(DFTRAMBYP_int)), (EMASA_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if ((AA_int >= WORDS) && (CENA_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
        XQA = 0 ? 1'b0 : 1'b1; QA_update = 0 ? 1'b0 : 1'b1;
    end else if (CENA_int === 1'b0 && (^AA_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if (DFTRAMBYP_int !== 1'b1) begin
      mux_address = (AA_int & 1'b1);
      row_address = (AA_int >> 1);
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      data_out = (row >> mux_address);
      mem_path = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
        	XQA = 1'b0; QA_update = 1'b1;
      end
        if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'b0) begin
        end else if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'bx) begin
        	XQA = 1'b1; QA_update = 1'b1;
        end
      if( isBitX(DFTRAMBYP_int) ) begin
        XQA = 1'b1; QA_update = 1'b1;
      end
      if( isBitX(SEA_int) && DFTRAMBYP_int === 1'b1 ) begin
        XQA = 1'b1; QA_update = 1'b1;
      end
      if(isBitX(DFTRAMBYP_int)) begin
        XQA = 1'b1; QA_update = 1'b1;
        failedWrite(0);
      end
    end
  end
  endtask

  task WriteB;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'bx) begin
      failedWrite(1);
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'b1) begin
      failedWrite(1);
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0 && (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAB_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) begin
      failedWrite(1);
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if ((AB_int >= WORDS) && (CENB_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
    end else if (CENB_int === 1'b0 && (^AB_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(1);
    end else if (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int))
        DB_int = {19{1'bx}};

      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int)) begin
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
      end
      mux_address = (AB_int & 1'b1);
      row_address = (AB_int >> 1);
      if (DFTRAMBYP_int !== 1'b1) begin
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      end
      if(isBitX(DFTRAMBYP_int)) begin
        writeEnable = {19{1'bx}};
        DB_int = {19{1'bx}};
      end else
          writeEnable = ~ {19{CENB_int}};
      row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
        1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
        1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
        1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
        1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
      new_data =  ( {1'b0, DB_int[18], 1'b0, DB_int[17], 1'b0, DB_int[16], 1'b0, DB_int[15],
        1'b0, DB_int[14], 1'b0, DB_int[13], 1'b0, DB_int[12], 1'b0, DB_int[11], 1'b0, DB_int[10],
        1'b0, DB_int[9], 1'b0, DB_int[8], 1'b0, DB_int[7], 1'b0, DB_int[6], 1'b0, DB_int[5],
        1'b0, DB_int[4], 1'b0, DB_int[3], 1'b0, DB_int[2], 1'b0, DB_int[1], 1'b0, DB_int[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        if (DFTRAMBYP_int === 1'b1 && (SEB_int === 1'b0 || SEB_int === 1'bx)) begin
        end else begin
        	mem[row_address] = row;
        end
    end
  end
  endtask
  always @ (CENA_ or TCENA_ or TENA_ or DFTRAMBYP_ or CLKA_) begin
  	if(CLKA_ == 1'b0) begin
  		CENA_p2 = CENA_;
  		TCENA_p2 = TCENA_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (VDDCE) begin
      if (VDDCE != 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDCE should be powered down after VDDPE, Illegal power down sequencing in %m at %0t", $time);
       end
        $display("In PowerDown Mode in %m at %0t", $time);
        failedWrite(0);
      end
      if (VDDCE == 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDPE should be powered up after VDDCE in %m at %0t", $time);
        $display("Illegal power up sequencing in %m at %0t", $time);
       end
        failedWrite(0);
      end
  end
`endif
`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_a == 1'b1 && (CENA_ === 1'bx || TCENA_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKA_ === 1'bx)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_a = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(0);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
        XQA = 1'b1; QA_update = 1'b1;
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_a == 1'b1) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
        XQA = 1'b1; QA_update = 1'b1;
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
    #0;
        QA_update = 1'b0;
  end

  always @ (CLKB_ or DFTRAMBYP_p2) begin
  	#0;
  	if(CLKB_ == 1'b1 && (DFTRAMBYP_int === 1'b1 || CENB_int != 1'b1)) begin
  	  if (RET1N_ == 1'b1) begin
	        DB_sh_update = 1'b1; 
  	  end
  	end
  end

  always @ CLKA_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKA_ === 1'bx || CLKA_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
        XQA = 1'b1; QA_update = 1'b1;
    end else if ((CLKA_ === 1'b1 || CLKA_ === 1'b0) && LAST_CLKA === 1'bx) begin
      XQA = 1'b0; QA_update = 1'b0; 
    end else if (CLKA_ === 1'b1 && LAST_CLKA === 1'b0) begin
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
  end else begin
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
  if (RET1N_ == 1'b1) begin
        XQA = 1'b0; QA_update = 1'b1;
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b0) begin
  if (RET1N_ == 1'b1) begin
        XQA = 1'b0; QA_update = 1'b1;
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else begin
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
    ReadA;
      if (CENA_int === 1'b0) previous_CLKA = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int, 1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
      end
  end
    end else if (CLKA_ === 1'b0 && LAST_CLKA === 1'b1) begin
      QA_update = 1'b0;
      XQA = 1'b0;
    end
  end
    LAST_CLKA = CLKA_;
  end

  reg globalNotifier0;
  initial globalNotifier0 = 1'b0;
  initial cont_flag0_int = 1'b0;

  always @ globalNotifier0 begin
    if ($realtime == 0) begin
    end else if ((EMAA_int[0] === 1'bx & DFTRAMBYP_int === 1'b1) || (EMAA_int[1] === 1'bx & DFTRAMBYP_int === 1'b1) || 
      (EMAA_int[2] === 1'bx & DFTRAMBYP_int === 1'b1) || (EMASA_int === 1'bx & DFTRAMBYP_int === 1'b1)
      ) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if ((CENA_int === 1'bx & DFTRAMBYP_int === 1'b0) || EMAA_int[0] === 1'bx || 
      EMAA_int[1] === 1'bx || EMAA_int[2] === 1'bx || EMASA_int === 1'bx || RET1N_int === 1'bx
       || clk0_int === 1'bx) begin
        XQA = 1'b1; QA_update = 1'b1;
    end else if (TENA_int === 1'bx) begin
      if(((CENA_ === 1'b1 & TCENA_ === 1'b1) & DFTRAMBYP_int === 1'b0) | (DFTRAMBYP_int === 1'b1 & SEA_int === 1'b1)) begin
      end else begin
      if (DFTRAMBYP_int === 1'b0) begin
        XQA = 1'b1; QA_update = 1'b1;
      end
      end
    end else if  (cont_flag0_int === 1'bx && COLLDISN_int === 1'b1 &&  (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
      AB_int, 1'b1, 1'b0)) begin
      cont_flag0_int = 1'b0;
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
    end else if  ((CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && cont_flag0_int === 1'bx && (COLLDISN_int === 1'b0 || COLLDISN_int === 
     1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
      cont_flag0_int = 1'b0;
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
    end else begin
      #0;
      ReadA;
   end
      #0;
        QA_update = 1'b0;
    globalNotifier0 = 1'b0;
  end

  assign SIA_int = SEA_ ? SIA_ : {2{1'b0}};

  datapath_latch_rf2_256x19_wm0 uDQA0 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[1]), .D(QA_int[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[0]), .XQ(XQA), .Q(QA_int[0]));
  datapath_latch_rf2_256x19_wm0 uDQA1 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[2]), .D(QA_int[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[1]), .XQ(XQA), .Q(QA_int[1]));
  datapath_latch_rf2_256x19_wm0 uDQA2 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[3]), .D(QA_int[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[2]), .XQ(XQA), .Q(QA_int[2]));
  datapath_latch_rf2_256x19_wm0 uDQA3 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[4]), .D(QA_int[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[3]), .XQ(XQA), .Q(QA_int[3]));
  datapath_latch_rf2_256x19_wm0 uDQA4 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[5]), .D(QA_int[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[4]), .XQ(XQA), .Q(QA_int[4]));
  datapath_latch_rf2_256x19_wm0 uDQA5 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[6]), .D(QA_int[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[5]), .XQ(XQA), .Q(QA_int[5]));
  datapath_latch_rf2_256x19_wm0 uDQA6 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[7]), .D(QA_int[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[6]), .XQ(XQA), .Q(QA_int[6]));
  datapath_latch_rf2_256x19_wm0 uDQA7 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[8]), .D(QA_int[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[7]), .XQ(XQA), .Q(QA_int[7]));
  datapath_latch_rf2_256x19_wm0 uDQA8 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(SIA_int[0]), .D(1'b0), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[8]), .XQ(XQA), .Q(QA_int[8]));
  datapath_latch_rf2_256x19_wm0 uDQA9 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(SIA_int[1]), .D(1'b0), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[9]), .XQ(XQA), .Q(QA_int[9]));
  datapath_latch_rf2_256x19_wm0 uDQA10 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[9]), .D(QA_int[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[10]), .XQ(XQA), .Q(QA_int[10]));
  datapath_latch_rf2_256x19_wm0 uDQA11 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[10]), .D(QA_int[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[11]), .XQ(XQA), .Q(QA_int[11]));
  datapath_latch_rf2_256x19_wm0 uDQA12 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[11]), .D(QA_int[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[12]), .XQ(XQA), .Q(QA_int[12]));
  datapath_latch_rf2_256x19_wm0 uDQA13 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[12]), .D(QA_int[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[13]), .XQ(XQA), .Q(QA_int[13]));
  datapath_latch_rf2_256x19_wm0 uDQA14 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[13]), .D(QA_int[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[14]), .XQ(XQA), .Q(QA_int[14]));
  datapath_latch_rf2_256x19_wm0 uDQA15 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[14]), .D(QA_int[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[15]), .XQ(XQA), .Q(QA_int[15]));
  datapath_latch_rf2_256x19_wm0 uDQA16 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[15]), .D(QA_int[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[16]), .XQ(XQA), .Q(QA_int[16]));
  datapath_latch_rf2_256x19_wm0 uDQA17 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[16]), .D(QA_int[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[17]), .XQ(XQA), .Q(QA_int[17]));
  datapath_latch_rf2_256x19_wm0 uDQA18 (.CLK(CLKA), .Q_update(QA_update), .SE(SEA_), .SI(QA_int[17]), .D(QA_int[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path[18]), .XQ(XQA), .Q(QA_int[18]));



  always @ (CENB_ or TCENB_ or TENB_ or DFTRAMBYP_ or CLKB_) begin
  	if(CLKB_ == 1'b0) begin
  		CENB_p2 = CENB_;
  		TCENB_p2 = TCENB_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_b == 1'b1 && (CENB_ === 1'bx || TCENB_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKB_ === 1'bx)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        XQA = 1'b1; QA_update = 1'b1;
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_b = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(1);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_b == 1'b1) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
    #0;
        QA_update = 1'b0;
        DB_sh_update = 1'b0; 
  end

  always @ CLKB_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKB_ === 1'bx || CLKB_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
    end else if ((CLKB_ === 1'b1 || CLKB_ === 1'b0) && LAST_CLKB === 1'bx) begin
       DB_sh_update = 1'b0;  XDB_sh = 1'b0;
    end else if (CLKB_ === 1'b1 && LAST_CLKB === 1'b0) begin
  if (RET1N_ == 1'b0) begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SEB_int = SEB_;
  end else begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SEB_int = SEB_;
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        XDB_sh = 1'b0; 
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
        XDB_sh = 1'b0; 
      end else begin
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        XDB_sh = 1'b0; 
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b0) begin
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
      end else begin
      WriteB;
      end
      if (CENB_int === 1'b0) previous_CLKB = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
    end
      end
    end else if (CLKB_ === 1'b0 && LAST_CLKB === 1'b1) begin
       DB_sh_update = 1'b0;  XDB_sh = 1'b0;
  end
  end
    LAST_CLKB = CLKB_;
  end

  reg globalNotifier1;
  initial globalNotifier1 = 1'b0;
  initial cont_flag1_int = 1'b0;

  always @ globalNotifier1 begin
    if ($realtime == 0) begin
    end else if ((EMAB_int[0] === 1'bx & DFTRAMBYP_int === 1'b1) || (EMAB_int[1] === 1'bx & DFTRAMBYP_int === 1'b1) || 
      (EMAB_int[2] === 1'bx & DFTRAMBYP_int === 1'b1)) begin
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if ((CENB_int === 1'bx & DFTRAMBYP_int === 1'b0) || EMAB_int[0] === 1'bx || 
      EMAB_int[1] === 1'bx || EMAB_int[2] === 1'bx || RET1N_int === 1'bx || clk1_int === 1'bx) begin
      failedWrite(1);
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
    end else if (TENB_int === 1'bx) begin
      if(((CENB_ === 1'b1 & TCENB_ === 1'b1) & DFTRAMBYP_int === 1'b0) | (DFTRAMBYP_int === 1'b1 & SEB_int === 1'b1)) begin
      end else begin
      if (DFTRAMBYP_int === 1'b0) begin
          failedWrite(1);
      end
        XDB_sh = 1'b1; 
        DB_sh_update = 1'b1; 
      end
    end else if  (cont_flag1_int === 1'bx && COLLDISN_int === 1'b1 &&  (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
      AB_int, 1'b1, 1'b0)) begin
      cont_flag1_int = 1'b0;
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
    end else if  ((CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && cont_flag1_int === 1'bx && (COLLDISN_int === 1'b0 || COLLDISN_int === 
     1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
      cont_flag1_int = 1'b0;
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        XQA = 1'b1; QA_update = 1'b1;
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
    end else begin
      #0;
      WriteB;
   end
      #0;
        DB_sh_update = 1'b0; 
    globalNotifier1 = 1'b0;
  end

  assign DB_int_bmux = TENB_ ? DB_ : TDB_;

  datapath_latch_rf2_256x19_wm0 uDQB0 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[1]), .D(DB_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[0]), .XQ(XDB_sh), .Q(DB_int_sh[0]));
  datapath_latch_rf2_256x19_wm0 uDQB1 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[2]), .D(DB_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[1]), .XQ(XDB_sh), .Q(DB_int_sh[1]));
  datapath_latch_rf2_256x19_wm0 uDQB2 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[3]), .D(DB_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[2]), .XQ(XDB_sh), .Q(DB_int_sh[2]));
  datapath_latch_rf2_256x19_wm0 uDQB3 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[4]), .D(DB_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[3]), .XQ(XDB_sh), .Q(DB_int_sh[3]));
  datapath_latch_rf2_256x19_wm0 uDQB4 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[5]), .D(DB_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[4]), .XQ(XDB_sh), .Q(DB_int_sh[4]));
  datapath_latch_rf2_256x19_wm0 uDQB5 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[6]), .D(DB_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[5]), .XQ(XDB_sh), .Q(DB_int_sh[5]));
  datapath_latch_rf2_256x19_wm0 uDQB6 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[7]), .D(DB_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[6]), .XQ(XDB_sh), .Q(DB_int_sh[6]));
  datapath_latch_rf2_256x19_wm0 uDQB7 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[8]), .D(DB_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[7]), .XQ(XDB_sh), .Q(DB_int_sh[7]));
  datapath_latch_rf2_256x19_wm0 uDQB8 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(SIB_[0]), .D(DB_int_bmux[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[8]), .XQ(XDB_sh), .Q(DB_int_sh[8]));
  datapath_latch_rf2_256x19_wm0 uDQB9 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(SIB_[1]), .D(DB_int_bmux[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[9]), .XQ(XDB_sh), .Q(DB_int_sh[9]));
  datapath_latch_rf2_256x19_wm0 uDQB10 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[9]), .D(DB_int_bmux[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[10]), .XQ(XDB_sh), .Q(DB_int_sh[10]));
  datapath_latch_rf2_256x19_wm0 uDQB11 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[10]), .D(DB_int_bmux[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[11]), .XQ(XDB_sh), .Q(DB_int_sh[11]));
  datapath_latch_rf2_256x19_wm0 uDQB12 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[11]), .D(DB_int_bmux[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[12]), .XQ(XDB_sh), .Q(DB_int_sh[12]));
  datapath_latch_rf2_256x19_wm0 uDQB13 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[12]), .D(DB_int_bmux[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[13]), .XQ(XDB_sh), .Q(DB_int_sh[13]));
  datapath_latch_rf2_256x19_wm0 uDQB14 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[13]), .D(DB_int_bmux[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[14]), .XQ(XDB_sh), .Q(DB_int_sh[14]));
  datapath_latch_rf2_256x19_wm0 uDQB15 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[14]), .D(DB_int_bmux[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[15]), .XQ(XDB_sh), .Q(DB_int_sh[15]));
  datapath_latch_rf2_256x19_wm0 uDQB16 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[15]), .D(DB_int_bmux[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[16]), .XQ(XDB_sh), .Q(DB_int_sh[16]));
  datapath_latch_rf2_256x19_wm0 uDQB17 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[16]), .D(DB_int_bmux[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[17]), .XQ(XDB_sh), .Q(DB_int_sh[17]));
  datapath_latch_rf2_256x19_wm0 uDQB18 (.CLK(CLKB), .Q_update(DB_sh_update), .SE(SEB_), .SI(DB_int_sh[17]), .D(DB_int_bmux[18]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(DB_int_bmux[18]), .XQ(XDB_sh), .Q(DB_int_sh[18]));



// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
 always @ (VDDCE or VDDPE or VSSE) begin
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
 end
`endif

  function row_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
    reg sameRow;
    reg sameMux;
    reg anyWrite;
  begin
    anyWrite = ((& wena) === 1'b1 && (& wenb) === 1'b1) ? 1'b0 : 1'b1;
    sameMux = (aa[0:0] == ab[0:0]) ? 1'b1 : 1'b0;
    if (aa[7:1] == ab[7:1]) begin
      sameRow = 1'b1;
    end else begin
      sameRow = 1'b0;
    end
    if (sameRow == 1'b1 && anyWrite == 1'b1)
      row_contention = 1'b1;
    else if (sameRow == 1'b1 && sameMux == 1'b1)
      row_contention = 1'b1;
    else
      row_contention = 1'b0;
  end
  endfunction

  function col_contention;
    input [7:0] aa;
    input [7:0] ab;
  begin
    if (aa[0:0] == ab[0:0])
      col_contention = 1'b1;
    else
      col_contention = 1'b0;
  end
  endfunction

  function is_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
  begin
    if ((& wena) === 1'b1 && (& wenb) === 1'b1) begin
      result = 1'b0;
    end else if (aa == ab) begin
      result = 1'b1;
    end else begin
      result = 1'b0;
    end
    is_contention = result;
  end
  endfunction

   wire contA_flag = (CENA_int !== 1'b1 && ((TENB_ ? CENB_ : TCENB_) !== 1'b1)) && ((COLLDISN_int === 1'b1 && is_contention(TENB_ ? AB_ : TAB_, AA_int, 1'b0, 1'b1)) ||
              ((COLLDISN_int === 1'b0 || COLLDISN_int === 1'bx) && row_contention(TENB_ ? AB_ : TAB_, AA_int, 1'b0, 1'b1)));
   wire contB_flag = (CENB_int !== 1'b1 && ((TENA_ ? CENA_ : TCENA_) !== 1'b1)) && ((COLLDISN_int === 1'b1 && is_contention(TENA_ ? AA_ : TAA_, AB_int, 1'b1, 1'b0)) ||
              ((COLLDISN_int === 1'b0 || COLLDISN_int === 1'bx) && row_contention(TENA_ ? AA_ : TAA_, AB_int, 1'b1, 1'b0)));

  always @ NOT_CENA begin
    CENA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA7 begin
    AA_int[7] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA6 begin
    AA_int[6] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA5 begin
    AA_int[5] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA4 begin
    AA_int[4] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA3 begin
    AA_int[3] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA2 begin
    AA_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA1 begin
    AA_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA0 begin
    AA_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CENB begin
    CENB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB7 begin
    AB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB6 begin
    AB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB5 begin
    AB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB4 begin
    AB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB3 begin
    AB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB2 begin
    AB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB1 begin
    AB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB0 begin
    AB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB18 begin
    DB_int[18] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB17 begin
    DB_int[17] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB16 begin
    DB_int[16] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB15 begin
    DB_int[15] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB14 begin
    DB_int[14] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB13 begin
    DB_int[13] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB12 begin
    DB_int[12] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB11 begin
    DB_int[11] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB10 begin
    DB_int[10] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB9 begin
    DB_int[9] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB8 begin
    DB_int[8] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB7 begin
    DB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB6 begin
    DB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB5 begin
    DB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB4 begin
    DB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB3 begin
    DB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB2 begin
    DB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB1 begin
    DB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB0 begin
    DB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_EMAA2 begin
    EMAA_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMAA1 begin
    EMAA_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMAA0 begin
    EMAA_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMASA begin
    EMASA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMAB2 begin
    EMAB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_EMAB1 begin
    EMAB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_EMAB0 begin
    EMAB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TENA begin
    TENA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TCENA begin
    CENA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA7 begin
    AA_int[7] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA6 begin
    AA_int[6] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA5 begin
    AA_int[5] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA4 begin
    AA_int[4] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA3 begin
    AA_int[3] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA2 begin
    AA_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA1 begin
    AA_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA0 begin
    AA_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TENB begin
    TENB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TCENB begin
    CENB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB7 begin
    AB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB6 begin
    AB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB5 begin
    AB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB4 begin
    AB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB3 begin
    AB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB2 begin
    AB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB1 begin
    AB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB0 begin
    AB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB18 begin
    DB_int[18] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB17 begin
    DB_int[17] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB16 begin
    DB_int[16] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB15 begin
    DB_int[15] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB14 begin
    DB_int[14] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB13 begin
    DB_int[13] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB12 begin
    DB_int[12] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB11 begin
    DB_int[11] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB10 begin
    DB_int[10] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB9 begin
    DB_int[9] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB8 begin
    DB_int[8] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB7 begin
    DB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB6 begin
    DB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB5 begin
    DB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB4 begin
    DB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB3 begin
    DB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB2 begin
    DB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB1 begin
    DB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB0 begin
    DB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_SIA1 begin
        XQA = 1'b1; QA_update = 1'b1;
  end
  always @ NOT_SIA0 begin
        XQA = 1'b1; QA_update = 1'b1;
  end
  always @ NOT_SEA begin
        XQA = 1'b1; QA_update = 1'b1;
    SEA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_DFTRAMBYP_CLKA begin
        XQA = 1'b1; QA_update = 1'b1;
    DFTRAMBYP_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_DFTRAMBYP_CLKB begin
        XDB_sh = 1'b1; DB_sh_update = 1'b1;
    DFTRAMBYP_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_RET1N begin
    RET1N_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_SIB1 begin
        XDB_sh = 1'b1; DB_sh_update = 1'b1;
  end
  always @ NOT_SIB0 begin
        XDB_sh = 1'b1; DB_sh_update = 1'b1;
  end
  always @ NOT_SEB begin
        XDB_sh = 1'b1; DB_sh_update = 1'b1;
        XDB_sh = 1'b1; DB_sh_update = 1'b1;
    SEB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_COLLDISN begin
    COLLDISN_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end

  always @ NOT_CONTA begin
    cont_flag0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CLKA_PER begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CLKA_MINH begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CLKA_MINL begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CONTB begin
    cont_flag1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_CLKB_PER begin
    clk1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_CLKB_MINH begin
    clk1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_CLKB_MINL begin
    clk1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end


  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq1;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq1;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq1;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq1;
  wire RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp;
  wire RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp;

  wire RET1Neq1aTENAeq1, RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0;
  wire RET1Neq1aTENBeq1, RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0;
  wire RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, RET1Neq1aTENAeq0;
  wire RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0;
  wire RET1Neq1aTENBeq0, RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0;
  wire RET1Neq1aSEAeq1, RET1Neq1aSEBeq1, RET1Neq1, RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp;
  wire RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp;

  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&!EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&!EMAA[1]&&EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&EMAA[1]&&EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&!EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&!EMAA[1]&&EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&EMAA[1]&&EMAA[0] && contA_flag;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&EMAA[0]&&EMASA;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&!EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&!EMAB[1]&&EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&EMAB[1]&&EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&!EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&!EMAB[1]&&EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&EMAB[1]&&EMAB[0] && contB_flag;
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&!EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&!EMAB[1]&&EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&EMAB[1]&&EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&!EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&!EMAB[1]&&EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&EMAB[1]&&EMAB[0];
  assign RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp = 
  RET1N&&TENB&&((DFTRAMBYP&&!SEB)||(!DFTRAMBYP&&!CENB));
  assign RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp = 
  RET1N&&(((TENA&&!CENA&&!DFTRAMBYP)||(!TENA&&!TCENA&&!DFTRAMBYP))||DFTRAMBYP);
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP);
  assign RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp = 
  RET1N&&!TENB&&((DFTRAMBYP&&!SEB)||(!TCENB&&!DFTRAMBYP));

  assign RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1 = RET1N&&TENA&&!CENA&&COLLDISN;
  assign RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0 = RET1N&&TENA&&!CENA&&!COLLDISN;
  assign RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1 = RET1N&&TENB&&!CENB&&COLLDISN;
  assign RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0 = RET1N&&TENB&&!CENB&&!COLLDISN;
  assign RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1 = RET1N&&!TENA&&!TCENA&&COLLDISN;
  assign RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0 = RET1N&&!TENA&&!TCENA&&!COLLDISN;
  assign RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1 = RET1N&&!TENB&&!TCENB&&COLLDISN;
  assign RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0 = RET1N&&!TENB&&!TCENB&&!COLLDISN;
  assign RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp = RET1N&&((TENA&&!CENA)||(!TENA&&!TCENA));
  assign RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp = RET1N&&((TENB&&!CENB)||(!TENB&&!TCENB));


  assign RET1Neq1aTENAeq1 = RET1N&&TENA;
  assign RET1Neq1aTENBeq1 = RET1N&&TENB;
  assign RET1Neq1aTENAeq0 = RET1N&&!TENA;
  assign RET1Neq1aTENBeq0 = RET1N&&!TENB;
  assign RET1Neq1aSEAeq1 = RET1N&&SEA;
  assign RET1Neq1aSEBeq1 = RET1N&&SEB;
  assign RET1Neq1 = RET1N;

  specify

    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (CENA +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TCENA +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENA == 1'b0 && CENA == 1'b1)
       (TENA +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENA == 1'b1 && CENA == 1'b0)
       (TENA -=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[7] +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[6] +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[5] +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[4] +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[3] +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[2] +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[1] +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[0] +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[7] +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[6] +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[5] +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[4] +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[3] +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[2] +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[1] +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[0] +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[7] == 1'b0 && AA[7] == 1'b1)
       (TENA +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[6] == 1'b0 && AA[6] == 1'b1)
       (TENA +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[5] == 1'b0 && AA[5] == 1'b1)
       (TENA +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[4] == 1'b0 && AA[4] == 1'b1)
       (TENA +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[3] == 1'b0 && AA[3] == 1'b1)
       (TENA +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[2] == 1'b0 && AA[2] == 1'b1)
       (TENA +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[1] == 1'b0 && AA[1] == 1'b1)
       (TENA +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[0] == 1'b0 && AA[0] == 1'b1)
       (TENA +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[7] == 1'b1 && AA[7] == 1'b0)
       (TENA -=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[6] == 1'b1 && AA[6] == 1'b0)
       (TENA -=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[5] == 1'b1 && AA[5] == 1'b0)
       (TENA -=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[4] == 1'b1 && AA[4] == 1'b0)
       (TENA -=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[3] == 1'b1 && AA[3] == 1'b0)
       (TENA -=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[2] == 1'b1 && AA[2] == 1'b0)
       (TENA -=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[1] == 1'b1 && AA[1] == 1'b0)
       (TENA -=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[0] == 1'b1 && AA[0] == 1'b0)
       (TENA -=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (CENB +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TCENB +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENB == 1'b0 && CENB == 1'b1)
       (TENB +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENB == 1'b1 && CENB == 1'b0)
       (TENB -=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[7] +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[6] +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[5] +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[4] +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[3] +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[2] +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[1] +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[0] +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[7] +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[6] +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[5] +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[4] +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[3] +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[2] +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[1] +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[0] +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[7] == 1'b0 && AB[7] == 1'b1)
       (TENB +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[6] == 1'b0 && AB[6] == 1'b1)
       (TENB +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[5] == 1'b0 && AB[5] == 1'b1)
       (TENB +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[4] == 1'b0 && AB[4] == 1'b1)
       (TENB +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[3] == 1'b0 && AB[3] == 1'b1)
       (TENB +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[2] == 1'b0 && AB[2] == 1'b1)
       (TENB +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[1] == 1'b0 && AB[1] == 1'b1)
       (TENB +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[0] == 1'b0 && AB[0] == 1'b1)
       (TENB +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[7] == 1'b1 && AB[7] == 1'b0)
       (TENB -=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[6] == 1'b1 && AB[6] == 1'b0)
       (TENB -=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[5] == 1'b1 && AB[5] == 1'b0)
       (TENB -=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[4] == 1'b1 && AB[4] == 1'b0)
       (TENB -=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[3] == 1'b1 && AB[3] == 1'b0)
       (TENB -=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[2] == 1'b1 && AB[2] == 1'b0)
       (TENB -=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[1] == 1'b1 && AB[1] == 1'b0)
       (TENB -=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[0] == 1'b1 && AB[0] == 1'b0)
       (TENB -=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (posedge CLKB => (SOB[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (posedge CLKB => (SOB[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);


   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $period(posedge CLKA, `ARM_MEM_PERIOD, NOT_CLKA_PER);
   `else
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
   `endif

   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $period(posedge CLKB, `ARM_MEM_PERIOD, NOT_CLKB_PER);
   `else
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
   `endif


   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $width(posedge CLKA, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINH);
       $width(negedge CLKA, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINL);
   `else
       $width(posedge CLKA &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINH);
       $width(negedge CLKA &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINL);
   `endif

   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $width(posedge CLKB, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINH);
       $width(negedge CLKB, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINL);
   `else
       $width(posedge CLKB &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINH);
       $width(negedge CLKB &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINL);
   `endif


    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);

    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);

    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1, posedge CENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1, negedge CENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1, posedge CENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1, negedge CENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB0);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMASA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMASA);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMASA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMASA);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB0);
    $setuphold(posedge CLKA &&& RET1Neq1, posedge TENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENA);
    $setuphold(posedge CLKA &&& RET1Neq1, negedge TENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0, posedge TCENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0, negedge TCENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKB &&& RET1Neq1, posedge TENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENB);
    $setuphold(posedge CLKB &&& RET1Neq1, negedge TENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0, posedge TCENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0, negedge TCENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB0);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, posedge SIA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA1);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, posedge SIA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA0);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, negedge SIA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA1);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, negedge SIA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA0);
    $setuphold(posedge CLKA &&& RET1Neq1, posedge SEA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEA);
    $setuphold(posedge CLKA &&& RET1Neq1, negedge SEA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEA);
    $setuphold(posedge CLKA &&& RET1Neq1, posedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKA);
    $setuphold(posedge CLKA &&& RET1Neq1, negedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKA);
    $setuphold(posedge CLKB &&& RET1Neq1, posedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKB);
    $setuphold(posedge CLKB &&& RET1Neq1, negedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKB);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, posedge SIB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB1);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, posedge SIB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB0);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, negedge SIB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB1);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, negedge SIB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB0);
    $setuphold(posedge CLKB &&& RET1Neq1, posedge SEB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEB);
    $setuphold(posedge CLKB &&& RET1Neq1, negedge SEB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEB);
    $setuphold(posedge CLKA &&& RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp, posedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(posedge CLKA &&& RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp, negedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(posedge CLKB &&& RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp, posedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(posedge CLKB &&& RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp, negedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(negedge RET1N, negedge CENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge CENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, negedge CENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge CENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, negedge TCENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge TCENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, negedge TCENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge TCENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge DFTRAMBYP, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge DFTRAMBYP, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENB, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENA, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENA, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENB, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENB, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENA, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENB, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENA, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, posedge DFTRAMBYP, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, posedge DFTRAMBYP, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
  endspecify


endmodule
`endcelldefine
  `endif
  `else
// If ARM_UD_MODEL is defined at Simulator Command Line, it Selects the Fast Functional Model
`ifdef ARM_UD_MODEL

// Following parameter Values can be overridden at Simulator Command Line.

// ARM_UD_DP Defines the delay through Data Paths, for Memory Models it represents BIST MUX output delays.
`ifdef ARM_UD_DP
`else
`define ARM_UD_DP #0.001
`endif
// ARM_UD_CP Defines the delay through Clock Path Cells, for Memory Models it is not used.
`ifdef ARM_UD_CP
`else
`define ARM_UD_CP
`endif
// ARM_UD_SEQ Defines the delay through the Memory, for Memory Models it is used for CLK->Q delays.
`ifdef ARM_UD_SEQ
`else
`define ARM_UD_SEQ #0.01
`endif

`celldefine
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module rf2_256x19_wm0 (VDDCE, VDDPE, VSSE, CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA,
    CENA, AA, CLKB, CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB,
    TAB, TDB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`else
module rf2_256x19_wm0 (CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA, CENA, AA, CLKB,
    CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB, TAB, TDB, RET1N,
    SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 19;
  parameter WORDS = 256;
  parameter MUX = 2;
  parameter MEM_WIDTH = 38; // redun block size 2, 18 on left, 20 on right
  parameter MEM_HEIGHT = 128;
  parameter WP_SIZE = 19 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 0;
  parameter UPMS_WIDTH = 1;

  output  CENYA;
  output [7:0] AYA;
  output  CENYB;
  output [7:0] AYB;
  output [18:0] QA;
  output [1:0] SOA;
  output [1:0] SOB;
  input  CLKA;
  input  CENA;
  input [7:0] AA;
  input  CLKB;
  input  CENB;
  input [7:0] AB;
  input [18:0] DB;
  input [2:0] EMAA;
  input  EMASA;
  input [2:0] EMAB;
  input  TENA;
  input  TCENA;
  input [7:0] TAA;
  input  TENB;
  input  TCENB;
  input [7:0] TAB;
  input [18:0] TDB;
  input  RET1N;
  input [1:0] SIA;
  input  SEA;
  input  DFTRAMBYP;
  input [1:0] SIB;
  input  SEB;
  input  COLLDISN;
`ifdef POWER_PINS
  inout VDDCE;
  inout VDDPE;
  inout VSSE;
`endif

  reg pre_charge_st;
  reg pre_charge_st_a;
  reg pre_charge_st_b;
  integer row_address;
  integer mux_address;
  initial row_address = 0;
  initial mux_address = 0;
  reg [37:0] mem [0:127];
  reg [37:0] row, row_t;
  reg LAST_CLKA;
  reg [37:0] row_mask;
  reg [37:0] new_data;
  reg [37:0] data_out;
  reg [18:0] readLatch0;
  reg [18:0] shifted_readLatch0;
  reg  read_mux_sel0_p2;
  reg [18:0] readLatch1;
  reg [18:0] shifted_readLatch1;
  reg  read_mux_sel1_p2;
  reg LAST_CLKB;
  reg [18:0] QA_int;
  reg [18:0] writeEnable;
  real previous_CLKA;
  real previous_CLKB;
  initial previous_CLKA = 0;
  initial previous_CLKB = 0;
  reg READ_WRITE, WRITE_WRITE, READ_READ, ROW_CC, COL_CC;
  reg READ_WRITE_1, WRITE_WRITE_1, READ_READ_1;
  reg  cont_flag0_int;
  reg  cont_flag1_int;
  initial cont_flag0_int = 1'b0;
  initial cont_flag1_int = 1'b0;
  reg clk0_int;
  reg clk1_int;

  wire  CENYA_;
  wire [7:0] AYA_;
  wire  CENYB_;
  wire [7:0] AYB_;
  wire [18:0] QA_;
  wire [1:0] SOA_;
  reg [1:0] SOA_int;
  wire [1:0] SOB_;
  reg [1:0] SOB_int;
 wire  CLKA_;
  wire  CENA_;
  reg  CENA_int;
  reg  CENA_p2;
  wire [7:0] AA_;
  reg [7:0] AA_int;
 wire  CLKB_;
  wire  CENB_;
  reg  CENB_int;
  reg  CENB_p2;
  wire [7:0] AB_;
  reg [7:0] AB_int;
  wire [18:0] DB_;
  reg [18:0] DB_int;
  reg [18:0] DB_int_sh;
  reg [18:0] DB_int_sh_int;
  wire [2:0] EMAA_;
  reg [2:0] EMAA_int;
  wire  EMASA_;
  reg  EMASA_int;
  wire [2:0] EMAB_;
  reg [2:0] EMAB_int;
  wire  TENA_;
  reg  TENA_int;
  wire  TCENA_;
  reg  TCENA_int;
  reg  TCENA_p2;
  wire [7:0] TAA_;
  reg [7:0] TAA_int;
  wire  TENB_;
  reg  TENB_int;
  wire  TCENB_;
  reg  TCENB_int;
  reg  TCENB_p2;
  wire [7:0] TAB_;
  reg [7:0] TAB_int;
  wire [18:0] TDB_;
  reg [18:0] TDB_int;
  wire  RET1N_;
  reg  RET1N_int;
  wire [1:0] SIA_;
  reg [1:0] SIA_int;
  wire  SEA_;
  reg  SEA_int;
  wire  DFTRAMBYP_;
  reg  DFTRAMBYP_int;
  reg  DFTRAMBYP_p2;
  wire [1:0] SIB_;
  reg [1:0] SIB_int;
  wire  SEB_;
  reg  SEB_int;
  wire  COLLDISN_;
  reg  COLLDISN_int;

  assign CENYA = CENYA_; 
  assign AYA[0] = AYA_[0]; 
  assign AYA[1] = AYA_[1]; 
  assign AYA[2] = AYA_[2]; 
  assign AYA[3] = AYA_[3]; 
  assign AYA[4] = AYA_[4]; 
  assign AYA[5] = AYA_[5]; 
  assign AYA[6] = AYA_[6]; 
  assign AYA[7] = AYA_[7]; 
  assign CENYB = CENYB_; 
  assign AYB[0] = AYB_[0]; 
  assign AYB[1] = AYB_[1]; 
  assign AYB[2] = AYB_[2]; 
  assign AYB[3] = AYB_[3]; 
  assign AYB[4] = AYB_[4]; 
  assign AYB[5] = AYB_[5]; 
  assign AYB[6] = AYB_[6]; 
  assign AYB[7] = AYB_[7]; 
  assign QA[0] = QA_[0]; 
  assign QA[1] = QA_[1]; 
  assign QA[2] = QA_[2]; 
  assign QA[3] = QA_[3]; 
  assign QA[4] = QA_[4]; 
  assign QA[5] = QA_[5]; 
  assign QA[6] = QA_[6]; 
  assign QA[7] = QA_[7]; 
  assign QA[8] = QA_[8]; 
  assign QA[9] = QA_[9]; 
  assign QA[10] = QA_[10]; 
  assign QA[11] = QA_[11]; 
  assign QA[12] = QA_[12]; 
  assign QA[13] = QA_[13]; 
  assign QA[14] = QA_[14]; 
  assign QA[15] = QA_[15]; 
  assign QA[16] = QA_[16]; 
  assign QA[17] = QA_[17]; 
  assign QA[18] = QA_[18]; 
  assign SOA[0] = SOA_[0]; 
  assign SOA[1] = SOA_[1]; 
  assign SOB[0] = SOB_[0]; 
  assign SOB[1] = SOB_[1]; 
  assign CLKA_ = CLKA;
  assign CENA_ = CENA;
  assign AA_[0] = AA[0];
  assign AA_[1] = AA[1];
  assign AA_[2] = AA[2];
  assign AA_[3] = AA[3];
  assign AA_[4] = AA[4];
  assign AA_[5] = AA[5];
  assign AA_[6] = AA[6];
  assign AA_[7] = AA[7];
  assign CLKB_ = CLKB;
  assign CENB_ = CENB;
  assign AB_[0] = AB[0];
  assign AB_[1] = AB[1];
  assign AB_[2] = AB[2];
  assign AB_[3] = AB[3];
  assign AB_[4] = AB[4];
  assign AB_[5] = AB[5];
  assign AB_[6] = AB[6];
  assign AB_[7] = AB[7];
  assign DB_[0] = DB[0];
  assign DB_[1] = DB[1];
  assign DB_[2] = DB[2];
  assign DB_[3] = DB[3];
  assign DB_[4] = DB[4];
  assign DB_[5] = DB[5];
  assign DB_[6] = DB[6];
  assign DB_[7] = DB[7];
  assign DB_[8] = DB[8];
  assign DB_[9] = DB[9];
  assign DB_[10] = DB[10];
  assign DB_[11] = DB[11];
  assign DB_[12] = DB[12];
  assign DB_[13] = DB[13];
  assign DB_[14] = DB[14];
  assign DB_[15] = DB[15];
  assign DB_[16] = DB[16];
  assign DB_[17] = DB[17];
  assign DB_[18] = DB[18];
  assign EMAA_[0] = EMAA[0];
  assign EMAA_[1] = EMAA[1];
  assign EMAA_[2] = EMAA[2];
  assign EMASA_ = EMASA;
  assign EMAB_[0] = EMAB[0];
  assign EMAB_[1] = EMAB[1];
  assign EMAB_[2] = EMAB[2];
  assign TENA_ = TENA;
  assign TCENA_ = TCENA;
  assign TAA_[0] = TAA[0];
  assign TAA_[1] = TAA[1];
  assign TAA_[2] = TAA[2];
  assign TAA_[3] = TAA[3];
  assign TAA_[4] = TAA[4];
  assign TAA_[5] = TAA[5];
  assign TAA_[6] = TAA[6];
  assign TAA_[7] = TAA[7];
  assign TENB_ = TENB;
  assign TCENB_ = TCENB;
  assign TAB_[0] = TAB[0];
  assign TAB_[1] = TAB[1];
  assign TAB_[2] = TAB[2];
  assign TAB_[3] = TAB[3];
  assign TAB_[4] = TAB[4];
  assign TAB_[5] = TAB[5];
  assign TAB_[6] = TAB[6];
  assign TAB_[7] = TAB[7];
  assign TDB_[0] = TDB[0];
  assign TDB_[1] = TDB[1];
  assign TDB_[2] = TDB[2];
  assign TDB_[3] = TDB[3];
  assign TDB_[4] = TDB[4];
  assign TDB_[5] = TDB[5];
  assign TDB_[6] = TDB[6];
  assign TDB_[7] = TDB[7];
  assign TDB_[8] = TDB[8];
  assign TDB_[9] = TDB[9];
  assign TDB_[10] = TDB[10];
  assign TDB_[11] = TDB[11];
  assign TDB_[12] = TDB[12];
  assign TDB_[13] = TDB[13];
  assign TDB_[14] = TDB[14];
  assign TDB_[15] = TDB[15];
  assign TDB_[16] = TDB[16];
  assign TDB_[17] = TDB[17];
  assign TDB_[18] = TDB[18];
  assign RET1N_ = RET1N;
  assign SIA_[0] = SIA[0];
  assign SIA_[1] = SIA[1];
  assign SEA_ = SEA;
  assign DFTRAMBYP_ = DFTRAMBYP;
  assign SIB_[0] = SIB[0];
  assign SIB_[1] = SIB[1];
  assign SEB_ = SEB;
  assign COLLDISN_ = COLLDISN;

  assign `ARM_UD_DP CENYA_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENA_ ? CENA_ : TCENA_)) : 1'bx;
  assign `ARM_UD_DP AYA_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENA_ ? AA_ : TAA_)) : {8{1'bx}};
  assign `ARM_UD_DP CENYB_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENB_ ? CENB_ : TCENB_)) : 1'bx;
  assign `ARM_UD_DP AYB_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENB_ ? AB_ : TAB_)) : {8{1'bx}};
  assign `ARM_UD_SEQ QA_ = (RET1N_ | pre_charge_st) ? ((QA_int)) : {19{1'bx}};
  assign `ARM_UD_DP SOA_ = (RET1N_ | pre_charge_st) ? ({QA_[18], QA_[0]}) : {2{1'bx}};
  assign `ARM_UD_DP SOB_ = (RET1N_ | pre_charge_st) ? (SOB_int) : {2{1'bx}};

// If INITIALIZE_MEMORY is defined at Simulator Command Line, it Initializes the Memory with all ZEROS.
`ifdef INITIALIZE_MEMORY
  integer i;
  initial begin
    #0;
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'b0}};
  end
`endif
  always @ (EMAA_) begin
  	if(EMAA_ < 3) 
   	$display("Warning: Set Value for EMAA doesn't match Default value 3 in %m at %0t", $time);
  end
  always @ (EMASA_) begin
  	if(EMASA_ < 0) 
   	$display("Warning: Set Value for EMASA doesn't match Default value 0 in %m at %0t", $time);
  end
  always @ (EMAB_) begin
  	if(EMAB_ < 3) 
   	$display("Warning: Set Value for EMAB doesn't match Default value 3 in %m at %0t", $time);
  end

  task failedWrite;
  input port_f;
  integer i;
  begin
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'bx}};
  end
  endtask

  function isBitX;
    input bitval;
    begin
      isBitX = ( bitval===1'bx || bitval===1'bz ) ? 1'b1 : 1'b0;
    end
  endfunction

  function isBit1;
    input bitval;
    begin
      isBit1 = ( bitval===1'b1 ) ? 1'b1 : 1'b0;
    end
  endfunction


task loadmem;
	input [1000*8-1:0] filename;
	reg [BITS-1:0] memld [0:WORDS-1];
	integer i;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	$readmemb(filename, memld);
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  wordtemp = memld[i];
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  	end
  end
  end
  endtask

task dumpmem;
	input [1000*8-1:0] filename_dump;
	integer i, dump_file_desc;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	dump_file_desc = $fopen(filename_dump, "w");
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      QA_int = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
   	$fdisplay(dump_file_desc, "%b", QA_int);
  end
  	end
    $fclose(dump_file_desc);
  end
  endtask

task loadaddr;
	input [7:0] load_addr;
	input [18:0] load_data;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  wordtemp = load_data;
	  Atemp = load_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  end
  end
  endtask

task dumpaddr;
	output [18:0] dump_data;
	input [7:0] dump_addr;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  Atemp = dump_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      QA_int = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
   	dump_data = QA_int;
  	end
  end
  endtask


  task ReadA;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'bx) begin
        QA_int = {19{1'bx}};
    end else if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'b1) begin
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0 && (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAA_int & isBit1(DFTRAMBYP_int)), (EMASA_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
        QA_int = {19{1'bx}};
    end else if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) begin
        QA_int = {19{1'bx}};
    end else if ((AA_int >= WORDS) && (CENA_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
      QA_int = 0 ? QA_int : {19{1'bx}};
    end else if (CENA_int === 1'b0 && (^AA_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if (DFTRAMBYP_int !== 1'b1) begin
      mux_address = (AA_int & 1'b1);
      row_address = (AA_int >> 1);
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      data_out = (row >> mux_address);
      QA_int = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
      end
        if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'b0) begin
        end else if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'bx) begin
             QA_int = {19{1'bx}};
        end
      if( isBitX(DFTRAMBYP_int) )
        QA_int = {19{1'bx}};
      if(isBitX(DFTRAMBYP_int)) begin
        QA_int = {19{1'bx}};
        failedWrite(0);
      end
    end
  end
  endtask

  task WriteB;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'bx) begin
      failedWrite(1);
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'b1) begin
      failedWrite(1);
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0 && (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAB_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) begin
      failedWrite(1);
  		#0;
           SOB_int = {2{1'bx}};
           DB_int_sh_int = {19{1'bx}};
           DB_int_sh = {19{1'bx}};
    end else if ((AB_int >= WORDS) && (CENB_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
    end else if (CENB_int === 1'b0 && (^AB_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(1);
    end else if (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int))
        DB_int = {19{1'bx}};

      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int)) begin
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
      end
      mux_address = (AB_int & 1'b1);
      row_address = (AB_int >> 1);
      if (DFTRAMBYP_int !== 1'b1) begin
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      end
      if(isBitX(DFTRAMBYP_int)) begin
        writeEnable = {19{1'bx}};
        DB_int = {19{1'bx}};
      end else
          writeEnable = ~ {19{CENB_int}};
      row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
        1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
        1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
        1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
        1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
      new_data =  ( {1'b0, DB_int[18], 1'b0, DB_int[17], 1'b0, DB_int[16], 1'b0, DB_int[15],
        1'b0, DB_int[14], 1'b0, DB_int[13], 1'b0, DB_int[12], 1'b0, DB_int[11], 1'b0, DB_int[10],
        1'b0, DB_int[9], 1'b0, DB_int[8], 1'b0, DB_int[7], 1'b0, DB_int[6], 1'b0, DB_int[5],
        1'b0, DB_int[4], 1'b0, DB_int[3], 1'b0, DB_int[2], 1'b0, DB_int[1], 1'b0, DB_int[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        if (DFTRAMBYP_int === 1'b1 && (SEB_int === 1'b0 || SEB_int === 1'bx)) begin
        end else begin
        	mem[row_address] = row;
        end
    end
  end
  endtask
  always @ (CENA_ or TCENA_ or TENA_ or DFTRAMBYP_ or CLKA_) begin
  	if(CLKA_ == 1'b0) begin
  		CENA_p2 = CENA_;
  		TCENA_p2 = TCENA_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (VDDCE) begin
      if (VDDCE != 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDCE should be powered down after VDDPE, Illegal power down sequencing in %m at %0t", $time);
       end
        $display("In PowerDown Mode in %m at %0t", $time);
        failedWrite(0);
      end
      if (VDDCE == 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDPE should be powered up after VDDCE in %m at %0t", $time);
        $display("Illegal power up sequencing in %m at %0t", $time);
       end
        failedWrite(0);
      end
  end
`endif
`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_a == 1'b1 && (CENA_ === 1'bx || TCENA_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKA_ === 1'bx)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_a = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(0);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
      QA_int = {19{1'bx}};
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SIA_int = {2{1'bx}};
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_a == 1'b1) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
        QA_int = {19{1'bx}};
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SIA_int = {2{1'bx}};
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
  end

  always @ (CLKB_ or DFTRAMBYP_) begin
  	#0;
  	if(CLKB_ == 1'b1 && (DFTRAMBYP_int === 1'b1 || CENB_int != 1'b1)) begin
  	  if (RET1N_ == 1'b1) begin
  		SOB_int = ({DB_int_sh[18], DB_int_sh[0]});
  		DB_int_sh_int = DB_int_sh;
  	  end
  	end
  end
  always @ (SIA_int) begin
  	#0;
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1 && ^SIA_int === 1'bx) begin
	QA_int[9] = SIA_int[1]; 
	QA_int[8] = SIA_int[0]; 
  	end
  end

  always @ CLKA_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKA_ === 1'bx || CLKA_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (CLKA_ === 1'b1 && LAST_CLKA === 1'b0) begin
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      SIA_int = SIA_;
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
  end else begin
      SIA_int = SIA_;
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      SIA_int = SIA_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
  if (RET1N_ == 1'b1) begin
	QA_int[18:9] = {QA_int[17:9], SIA_[1]}; 
	QA_int[8:0] = {SIA_[0], QA_int[8:1]}; 
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b0) begin
  if (RET1N_ == 1'b1) begin
	QA_int[18:9] = {QA_int[17:9], 1'b0}; 
	QA_int[8:0] = {1'b0, QA_int[8:1]}; 
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else begin
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      SIA_int = SIA_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
    ReadA;
      if (CENA_int === 1'b0) previous_CLKA = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int, 1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
      end
  end
    end else if (CLKA_ === 1'b0 && LAST_CLKA === 1'b1) begin
    end
  end
    LAST_CLKA = CLKA_;
  end
  always @ (CENB_ or TCENB_ or TENB_ or DFTRAMBYP_ or CLKB_) begin
  	if(CLKB_ == 1'b0) begin
  		CENB_p2 = CENB_;
  		TCENB_p2 = TCENB_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_b == 1'b1 && (CENB_ === 1'bx || TCENB_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKB_ === 1'bx)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_b = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(1);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
      DB_int_sh = {19{1'bx}};
      DB_int_sh_int = {19{1'bx}};
      SOB_int = {2{1'bx}};
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SIB_int = {2{1'bx}};
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_b == 1'b1) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
      DB_int_sh = {19{1'bx}};
      DB_int_sh_int = {19{1'bx}};
      SOB_int = {2{1'bx}};
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SIB_int = {2{1'bx}};
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
  end

  always @ (SIB_int) begin
  	#0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1 && ^SIB_int === 1'bx) begin
	DB_int_sh_int[9] = SIB_int[1]; 
	DB_int_sh_int[8] = SIB_int[0]; 
  	end
  end
  always @ CLKB_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKB_ === 1'bx || CLKB_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
    end else if (CLKB_ === 1'b1 && LAST_CLKB === 1'b0) begin
  if (RET1N_ == 1'b0) begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SIB_int = SIB_;
      SEB_int = SEB_;
  end else begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SIB_int = SIB_;
      SEB_int = SEB_;
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      SIB_int = SIB_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        DB_int_sh = TENB_ ? DB_ : TDB_;
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
	DB_int_sh[18:9] = {DB_int_sh_int[17:9], SIB_[1]}; 
	DB_int_sh[8:0] = {SIB_[0], DB_int_sh_int[8:1]}; 
      end else begin
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      SIB_int = SIB_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        DB_int_sh_int = DB_int_sh;
        DB_int_sh = TENB_ ? DB_ : TDB_;
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b0) begin
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
      end else begin
      WriteB;
      end
      if (CENB_int === 1'b0) previous_CLKB = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
    end
      end
  end
  end
    LAST_CLKB = CLKB_;
  end
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
 always @ (VDDCE or VDDPE or VSSE) begin
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
 end
`endif

  function row_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
    reg sameRow;
    reg sameMux;
    reg anyWrite;
  begin
    anyWrite = ((& wena) === 1'b1 && (& wenb) === 1'b1) ? 1'b0 : 1'b1;
    sameMux = (aa[0:0] == ab[0:0]) ? 1'b1 : 1'b0;
    if (aa[7:1] == ab[7:1]) begin
      sameRow = 1'b1;
    end else begin
      sameRow = 1'b0;
    end
    if (sameRow == 1'b1 && anyWrite == 1'b1)
      row_contention = 1'b1;
    else if (sameRow == 1'b1 && sameMux == 1'b1)
      row_contention = 1'b1;
    else
      row_contention = 1'b0;
  end
  endfunction

  function col_contention;
    input [7:0] aa;
    input [7:0] ab;
  begin
    if (aa[0:0] == ab[0:0])
      col_contention = 1'b1;
    else
      col_contention = 1'b0;
  end
  endfunction

  function is_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
  begin
    if ((& wena) === 1'b1 && (& wenb) === 1'b1) begin
      result = 1'b0;
    end else if (aa == ab) begin
      result = 1'b1;
    end else begin
      result = 1'b0;
    end
    is_contention = result;
  end
  endfunction


endmodule
`endcelldefine
`else
`celldefine
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module rf2_256x19_wm0 (VDDCE, VDDPE, VSSE, CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA,
    CENA, AA, CLKB, CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB,
    TAB, TDB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`else
module rf2_256x19_wm0 (CENYA, AYA, CENYB, AYB, QA, SOA, SOB, CLKA, CENA, AA, CLKB,
    CENB, AB, DB, EMAA, EMASA, EMAB, TENA, TCENA, TAA, TENB, TCENB, TAB, TDB, RET1N,
    SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 19;
  parameter WORDS = 256;
  parameter MUX = 2;
  parameter MEM_WIDTH = 38; // redun block size 2, 18 on left, 20 on right
  parameter MEM_HEIGHT = 128;
  parameter WP_SIZE = 19 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 0;
  parameter UPMS_WIDTH = 1;

  output  CENYA;
  output [7:0] AYA;
  output  CENYB;
  output [7:0] AYB;
  output [18:0] QA;
  output [1:0] SOA;
  output [1:0] SOB;
  input  CLKA;
  input  CENA;
  input [7:0] AA;
  input  CLKB;
  input  CENB;
  input [7:0] AB;
  input [18:0] DB;
  input [2:0] EMAA;
  input  EMASA;
  input [2:0] EMAB;
  input  TENA;
  input  TCENA;
  input [7:0] TAA;
  input  TENB;
  input  TCENB;
  input [7:0] TAB;
  input [18:0] TDB;
  input  RET1N;
  input [1:0] SIA;
  input  SEA;
  input  DFTRAMBYP;
  input [1:0] SIB;
  input  SEB;
  input  COLLDISN;
`ifdef POWER_PINS
  inout VDDCE;
  inout VDDPE;
  inout VSSE;
`endif

  reg pre_charge_st;
  reg pre_charge_st_a;
  reg pre_charge_st_b;
  integer row_address;
  integer mux_address;
  initial row_address = 0;
  initial mux_address = 0;
  reg [37:0] mem [0:127];
  reg [37:0] row, row_t;
  reg LAST_CLKA;
  reg [37:0] row_mask;
  reg [37:0] new_data;
  reg [37:0] data_out;
  reg [18:0] readLatch0;
  reg [18:0] shifted_readLatch0;
  reg  read_mux_sel0_p2;
  reg [18:0] readLatch1;
  reg [18:0] shifted_readLatch1;
  reg  read_mux_sel1_p2;
  reg LAST_CLKB;
  reg [18:0] QA_int;
  reg [18:0] writeEnable;
  real previous_CLKA;
  real previous_CLKB;
  initial previous_CLKA = 0;
  initial previous_CLKB = 0;
  reg READ_WRITE, WRITE_WRITE, READ_READ, ROW_CC, COL_CC;
  reg READ_WRITE_1, WRITE_WRITE_1, READ_READ_1;
  reg  cont_flag0_int;
  reg  cont_flag1_int;
  initial cont_flag0_int = 1'b0;
  initial cont_flag1_int = 1'b0;

  reg NOT_CENA, NOT_AA7, NOT_AA6, NOT_AA5, NOT_AA4, NOT_AA3, NOT_AA2, NOT_AA1, NOT_AA0;
  reg NOT_CENB, NOT_AB7, NOT_AB6, NOT_AB5, NOT_AB4, NOT_AB3, NOT_AB2, NOT_AB1, NOT_AB0;
  reg NOT_DB18, NOT_DB17, NOT_DB16, NOT_DB15, NOT_DB14, NOT_DB13, NOT_DB12, NOT_DB11;
  reg NOT_DB10, NOT_DB9, NOT_DB8, NOT_DB7, NOT_DB6, NOT_DB5, NOT_DB4, NOT_DB3, NOT_DB2;
  reg NOT_DB1, NOT_DB0, NOT_EMAA2, NOT_EMAA1, NOT_EMAA0, NOT_EMASA, NOT_EMAB2, NOT_EMAB1;
  reg NOT_EMAB0, NOT_TENA, NOT_TCENA, NOT_TAA7, NOT_TAA6, NOT_TAA5, NOT_TAA4, NOT_TAA3;
  reg NOT_TAA2, NOT_TAA1, NOT_TAA0, NOT_TENB, NOT_TCENB, NOT_TAB7, NOT_TAB6, NOT_TAB5;
  reg NOT_TAB4, NOT_TAB3, NOT_TAB2, NOT_TAB1, NOT_TAB0, NOT_TDB18, NOT_TDB17, NOT_TDB16;
  reg NOT_TDB15, NOT_TDB14, NOT_TDB13, NOT_TDB12, NOT_TDB11, NOT_TDB10, NOT_TDB9, NOT_TDB8;
  reg NOT_TDB7, NOT_TDB6, NOT_TDB5, NOT_TDB4, NOT_TDB3, NOT_TDB2, NOT_TDB1, NOT_TDB0;
  reg NOT_SIA1, NOT_SIA0, NOT_SEA, NOT_DFTRAMBYP_CLKA, NOT_DFTRAMBYP_CLKB, NOT_RET1N;
  reg NOT_SIB1, NOT_SIB0, NOT_SEB, NOT_COLLDISN;
  reg NOT_CONTA, NOT_CLKA_PER, NOT_CLKA_MINH, NOT_CLKA_MINL, NOT_CONTB, NOT_CLKB_PER;
  reg NOT_CLKB_MINH, NOT_CLKB_MINL;
  reg clk0_int;
  reg clk1_int;

  wire  CENYA_;
  wire [7:0] AYA_;
  wire  CENYB_;
  wire [7:0] AYB_;
  wire [18:0] QA_;
  wire [1:0] SOA_;
  reg [1:0] SOA_int;
  wire [1:0] SOB_;
  reg [1:0] SOB_int;
 wire  CLKA_;
  wire  CENA_;
  reg  CENA_int;
  reg  CENA_p2;
  wire [7:0] AA_;
  reg [7:0] AA_int;
 wire  CLKB_;
  wire  CENB_;
  reg  CENB_int;
  reg  CENB_p2;
  wire [7:0] AB_;
  reg [7:0] AB_int;
  wire [18:0] DB_;
  reg [18:0] DB_int;
  reg [18:0] DB_int_sh;
  reg [18:0] DB_int_sh_int;
  wire [2:0] EMAA_;
  reg [2:0] EMAA_int;
  wire  EMASA_;
  reg  EMASA_int;
  wire [2:0] EMAB_;
  reg [2:0] EMAB_int;
  wire  TENA_;
  reg  TENA_int;
  wire  TCENA_;
  reg  TCENA_int;
  reg  TCENA_p2;
  wire [7:0] TAA_;
  reg [7:0] TAA_int;
  wire  TENB_;
  reg  TENB_int;
  wire  TCENB_;
  reg  TCENB_int;
  reg  TCENB_p2;
  wire [7:0] TAB_;
  reg [7:0] TAB_int;
  wire [18:0] TDB_;
  reg [18:0] TDB_int;
  wire  RET1N_;
  reg  RET1N_int;
  wire [1:0] SIA_;
  reg [1:0] SIA_int;
  wire  SEA_;
  reg  SEA_int;
  wire  DFTRAMBYP_;
  reg  DFTRAMBYP_int;
  reg  DFTRAMBYP_p2;
  wire [1:0] SIB_;
  reg [1:0] SIB_int;
  wire  SEB_;
  reg  SEB_int;
  wire  COLLDISN_;
  reg  COLLDISN_int;

  buf B135(CENYA, CENYA_);
  buf B136(AYA[0], AYA_[0]);
  buf B137(AYA[1], AYA_[1]);
  buf B138(AYA[2], AYA_[2]);
  buf B139(AYA[3], AYA_[3]);
  buf B140(AYA[4], AYA_[4]);
  buf B141(AYA[5], AYA_[5]);
  buf B142(AYA[6], AYA_[6]);
  buf B143(AYA[7], AYA_[7]);
  buf B144(CENYB, CENYB_);
  buf B145(AYB[0], AYB_[0]);
  buf B146(AYB[1], AYB_[1]);
  buf B147(AYB[2], AYB_[2]);
  buf B148(AYB[3], AYB_[3]);
  buf B149(AYB[4], AYB_[4]);
  buf B150(AYB[5], AYB_[5]);
  buf B151(AYB[6], AYB_[6]);
  buf B152(AYB[7], AYB_[7]);
  buf B153(QA[0], QA_[0]);
  buf B154(QA[1], QA_[1]);
  buf B155(QA[2], QA_[2]);
  buf B156(QA[3], QA_[3]);
  buf B157(QA[4], QA_[4]);
  buf B158(QA[5], QA_[5]);
  buf B159(QA[6], QA_[6]);
  buf B160(QA[7], QA_[7]);
  buf B161(QA[8], QA_[8]);
  buf B162(QA[9], QA_[9]);
  buf B163(QA[10], QA_[10]);
  buf B164(QA[11], QA_[11]);
  buf B165(QA[12], QA_[12]);
  buf B166(QA[13], QA_[13]);
  buf B167(QA[14], QA_[14]);
  buf B168(QA[15], QA_[15]);
  buf B169(QA[16], QA_[16]);
  buf B170(QA[17], QA_[17]);
  buf B171(QA[18], QA_[18]);
  buf B172(SOA[0], SOA_[0]);
  buf B173(SOA[1], SOA_[1]);
  buf B174(SOB[0], SOB_[0]);
  buf B175(SOB[1], SOB_[1]);
  buf B176(CLKA_, CLKA);
  buf B177(CENA_, CENA);
  buf B178(AA_[0], AA[0]);
  buf B179(AA_[1], AA[1]);
  buf B180(AA_[2], AA[2]);
  buf B181(AA_[3], AA[3]);
  buf B182(AA_[4], AA[4]);
  buf B183(AA_[5], AA[5]);
  buf B184(AA_[6], AA[6]);
  buf B185(AA_[7], AA[7]);
  buf B186(CLKB_, CLKB);
  buf B187(CENB_, CENB);
  buf B188(AB_[0], AB[0]);
  buf B189(AB_[1], AB[1]);
  buf B190(AB_[2], AB[2]);
  buf B191(AB_[3], AB[3]);
  buf B192(AB_[4], AB[4]);
  buf B193(AB_[5], AB[5]);
  buf B194(AB_[6], AB[6]);
  buf B195(AB_[7], AB[7]);
  buf B196(DB_[0], DB[0]);
  buf B197(DB_[1], DB[1]);
  buf B198(DB_[2], DB[2]);
  buf B199(DB_[3], DB[3]);
  buf B200(DB_[4], DB[4]);
  buf B201(DB_[5], DB[5]);
  buf B202(DB_[6], DB[6]);
  buf B203(DB_[7], DB[7]);
  buf B204(DB_[8], DB[8]);
  buf B205(DB_[9], DB[9]);
  buf B206(DB_[10], DB[10]);
  buf B207(DB_[11], DB[11]);
  buf B208(DB_[12], DB[12]);
  buf B209(DB_[13], DB[13]);
  buf B210(DB_[14], DB[14]);
  buf B211(DB_[15], DB[15]);
  buf B212(DB_[16], DB[16]);
  buf B213(DB_[17], DB[17]);
  buf B214(DB_[18], DB[18]);
  buf B215(EMAA_[0], EMAA[0]);
  buf B216(EMAA_[1], EMAA[1]);
  buf B217(EMAA_[2], EMAA[2]);
  buf B218(EMASA_, EMASA);
  buf B219(EMAB_[0], EMAB[0]);
  buf B220(EMAB_[1], EMAB[1]);
  buf B221(EMAB_[2], EMAB[2]);
  buf B222(TENA_, TENA);
  buf B223(TCENA_, TCENA);
  buf B224(TAA_[0], TAA[0]);
  buf B225(TAA_[1], TAA[1]);
  buf B226(TAA_[2], TAA[2]);
  buf B227(TAA_[3], TAA[3]);
  buf B228(TAA_[4], TAA[4]);
  buf B229(TAA_[5], TAA[5]);
  buf B230(TAA_[6], TAA[6]);
  buf B231(TAA_[7], TAA[7]);
  buf B232(TENB_, TENB);
  buf B233(TCENB_, TCENB);
  buf B234(TAB_[0], TAB[0]);
  buf B235(TAB_[1], TAB[1]);
  buf B236(TAB_[2], TAB[2]);
  buf B237(TAB_[3], TAB[3]);
  buf B238(TAB_[4], TAB[4]);
  buf B239(TAB_[5], TAB[5]);
  buf B240(TAB_[6], TAB[6]);
  buf B241(TAB_[7], TAB[7]);
  buf B242(TDB_[0], TDB[0]);
  buf B243(TDB_[1], TDB[1]);
  buf B244(TDB_[2], TDB[2]);
  buf B245(TDB_[3], TDB[3]);
  buf B246(TDB_[4], TDB[4]);
  buf B247(TDB_[5], TDB[5]);
  buf B248(TDB_[6], TDB[6]);
  buf B249(TDB_[7], TDB[7]);
  buf B250(TDB_[8], TDB[8]);
  buf B251(TDB_[9], TDB[9]);
  buf B252(TDB_[10], TDB[10]);
  buf B253(TDB_[11], TDB[11]);
  buf B254(TDB_[12], TDB[12]);
  buf B255(TDB_[13], TDB[13]);
  buf B256(TDB_[14], TDB[14]);
  buf B257(TDB_[15], TDB[15]);
  buf B258(TDB_[16], TDB[16]);
  buf B259(TDB_[17], TDB[17]);
  buf B260(TDB_[18], TDB[18]);
  buf B261(RET1N_, RET1N);
  buf B262(SIA_[0], SIA[0]);
  buf B263(SIA_[1], SIA[1]);
  buf B264(SEA_, SEA);
  buf B265(DFTRAMBYP_, DFTRAMBYP);
  buf B266(SIB_[0], SIB[0]);
  buf B267(SIB_[1], SIB[1]);
  buf B268(SEB_, SEB);
  buf B269(COLLDISN_, COLLDISN);

  assign CENYA_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENA_ ? CENA_ : TCENA_)) : 1'bx;
  assign AYA_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENA_ ? AA_ : TAA_)) : {8{1'bx}};
  assign CENYB_ = (RET1N_ | pre_charge_st) ? (DFTRAMBYP_ & (TENB_ ? CENB_ : TCENB_)) : 1'bx;
  assign AYB_ = (RET1N_ | pre_charge_st) ? ({8{DFTRAMBYP_}} & (TENB_ ? AB_ : TAB_)) : {8{1'bx}};
   `ifdef ARM_FAULT_MODELING
     rf2_256x19_wm0_error_injection u1(.CLK(CLKA_), .Q_out(QA_), .A(AA_int), .CEN(CENA_int), .DFTRAMBYP(DFTRAMBYP_int), .SE(SEA_int), .Q_in(QA_int));
  `else
  assign QA_ = (RET1N_ | pre_charge_st) ? ((QA_int)) : {19{1'bx}};
  `endif
  assign SOA_ = (RET1N_ | pre_charge_st) ? ({QA_[18], QA_[0]}) : {2{1'bx}};
  assign SOB_ = (RET1N_ | pre_charge_st) ? (SOB_int) : {2{1'bx}};

// If INITIALIZE_MEMORY is defined at Simulator Command Line, it Initializes the Memory with all ZEROS.
`ifdef INITIALIZE_MEMORY
  integer i;
  initial begin
    #0;
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'b0}};
  end
`endif
  always @ (EMAA_) begin
  	if(EMAA_ < 3) 
   	$display("Warning: Set Value for EMAA doesn't match Default value 3 in %m at %0t", $time);
  end
  always @ (EMASA_) begin
  	if(EMASA_ < 0) 
   	$display("Warning: Set Value for EMASA doesn't match Default value 0 in %m at %0t", $time);
  end
  always @ (EMAB_) begin
  	if(EMAB_ < 3) 
   	$display("Warning: Set Value for EMAB doesn't match Default value 3 in %m at %0t", $time);
  end

  task failedWrite;
  input port_f;
  integer i;
  begin
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'bx}};
  end
  endtask

  function isBitX;
    input bitval;
    begin
      isBitX = ( bitval===1'bx || bitval===1'bz ) ? 1'b1 : 1'b0;
    end
  endfunction

  function isBit1;
    input bitval;
    begin
      isBit1 = ( bitval===1'b1 ) ? 1'b1 : 1'b0;
    end
  endfunction


task loadmem;
	input [1000*8-1:0] filename;
	reg [BITS-1:0] memld [0:WORDS-1];
	integer i;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	$readmemb(filename, memld);
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  wordtemp = memld[i];
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  	end
  end
  end
  endtask

task dumpmem;
	input [1000*8-1:0] filename_dump;
	integer i, dump_file_desc;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
	dump_file_desc = $fopen(filename_dump, "w");
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  for (i=0;i<WORDS;i=i+1) begin
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      QA_int = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
   	$fdisplay(dump_file_desc, "%b", QA_int);
  end
  	end
    $fclose(dump_file_desc);
  end
  endtask

task loadaddr;
	input [7:0] load_addr;
	input [18:0] load_data;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  wordtemp = load_data;
	  Atemp = load_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
        row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
          1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
          1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
          1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
          1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[18], 1'b0, wordtemp[17], 1'b0, wordtemp[16],
          1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13], 1'b0, wordtemp[12],
          1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9], 1'b0, wordtemp[8],
          1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5], 1'b0, wordtemp[4],
          1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1], 1'b0, wordtemp[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
  end
  end
  endtask

task dumpaddr;
	output [18:0] dump_data;
	input [7:0] dump_addr;
	reg [BITS-1:0] wordtemp;
	reg [7:0] Atemp;
  begin
     if (CENA_ === 1'b1 && CENB_ === 1'b1) begin
	  Atemp = dump_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {19{1'b1}};
      data_out = (row >> mux_address);
      QA_int = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
   	dump_data = QA_int;
  	end
  end
  endtask


  task ReadA;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'bx) begin
        QA_int = {19{1'bx}};
    end else if (DFTRAMBYP_int=== 1'b0 && SEA_int === 1'b1) begin
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0 && (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAA_int & isBit1(DFTRAMBYP_int)), (EMASA_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
        QA_int = {19{1'bx}};
    end else if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) begin
        QA_int = {19{1'bx}};
    end else if ((AA_int >= WORDS) && (CENA_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
      QA_int = 0 ? QA_int : {19{1'bx}};
    end else if (CENA_int === 1'b0 && (^AA_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (CENA_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if (DFTRAMBYP_int !== 1'b1) begin
      mux_address = (AA_int & 1'b1);
      row_address = (AA_int >> 1);
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      data_out = (row >> mux_address);
      QA_int = {data_out[36], data_out[34], data_out[32], data_out[30], data_out[28],
        data_out[26], data_out[24], data_out[22], data_out[20], data_out[18], data_out[16],
        data_out[14], data_out[12], data_out[10], data_out[8], data_out[6], data_out[4],
        data_out[2], data_out[0]};
      end
        if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'b0) begin
        end else if (DFTRAMBYP_int === 1'b1 && SEA_int === 1'bx) begin
             QA_int = {19{1'bx}};
        end
      if( isBitX(DFTRAMBYP_int) )
        QA_int = {19{1'bx}};
      if(isBitX(DFTRAMBYP_int)) begin
        QA_int = {19{1'bx}};
        failedWrite(0);
      end
    end
  end
  endtask

  task WriteB;
  begin
    if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'bx) begin
      failedWrite(1);
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if (DFTRAMBYP_int=== 1'b0 && SEB_int === 1'b1) begin
      failedWrite(1);
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if (RET1N_int === 1'bx || RET1N_int === 1'bz) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0 && (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{(EMAB_int & isBit1(DFTRAMBYP_int))} === 1'bx) begin
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) begin
      failedWrite(1);
  		#0;
           SOB_int = {2{1'bx}};
           DB_int_sh_int = {19{1'bx}};
           DB_int_sh = {19{1'bx}};
    end else if ((AB_int >= WORDS) && (CENB_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
    end else if (CENB_int === 1'b0 && (^AB_int) === 1'bx && DFTRAMBYP_int === 1'b0) begin
      failedWrite(1);
    end else if (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int))
        DB_int = {19{1'bx}};

      if(isBitX(DFTRAMBYP_int) || isBitX(SEB_int)) begin
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
      end
      mux_address = (AB_int & 1'b1);
      row_address = (AB_int >> 1);
      if (DFTRAMBYP_int !== 1'b1) begin
      if (row_address > 127)
        row = {38{1'bx}};
      else
        row = mem[row_address];
      end
      if(isBitX(DFTRAMBYP_int)) begin
        writeEnable = {19{1'bx}};
        DB_int = {19{1'bx}};
      end else
          writeEnable = ~ {19{CENB_int}};
      row_mask =  ( {1'b0, writeEnable[18], 1'b0, writeEnable[17], 1'b0, writeEnable[16],
        1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13], 1'b0, writeEnable[12],
        1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9], 1'b0, writeEnable[8],
        1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5], 1'b0, writeEnable[4],
        1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1], 1'b0, writeEnable[0]} << mux_address);
      new_data =  ( {1'b0, DB_int[18], 1'b0, DB_int[17], 1'b0, DB_int[16], 1'b0, DB_int[15],
        1'b0, DB_int[14], 1'b0, DB_int[13], 1'b0, DB_int[12], 1'b0, DB_int[11], 1'b0, DB_int[10],
        1'b0, DB_int[9], 1'b0, DB_int[8], 1'b0, DB_int[7], 1'b0, DB_int[6], 1'b0, DB_int[5],
        1'b0, DB_int[4], 1'b0, DB_int[3], 1'b0, DB_int[2], 1'b0, DB_int[1], 1'b0, DB_int[0]} << mux_address);
      row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        if (DFTRAMBYP_int === 1'b1 && (SEB_int === 1'b0 || SEB_int === 1'bx)) begin
        end else begin
        	mem[row_address] = row;
        end
    end
  end
  endtask
  always @ (CENA_ or TCENA_ or TENA_ or DFTRAMBYP_ or CLKA_) begin
  	if(CLKA_ == 1'b0) begin
  		CENA_p2 = CENA_;
  		TCENA_p2 = TCENA_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (VDDCE) begin
      if (VDDCE != 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDCE should be powered down after VDDPE, Illegal power down sequencing in %m at %0t", $time);
       end
        $display("In PowerDown Mode in %m at %0t", $time);
        failedWrite(0);
      end
      if (VDDCE == 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDPE should be powered up after VDDCE in %m at %0t", $time);
        $display("Illegal power up sequencing in %m at %0t", $time);
       end
        failedWrite(0);
      end
  end
`endif
`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_a == 1'b1 && (CENA_ === 1'bx || TCENA_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKA_ === 1'bx)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENA_p2 === 1'b0 || TCENA_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_a = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(0);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
      QA_int = {19{1'bx}};
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SIA_int = {2{1'bx}};
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_a == 1'b1) begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_a = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
        QA_int = {19{1'bx}};
      CENA_int = 1'bx;
      AA_int = {8{1'bx}};
      EMAA_int = {3{1'bx}};
      EMASA_int = 1'bx;
      TENA_int = 1'bx;
      TCENA_int = 1'bx;
      TAA_int = {8{1'bx}};
      RET1N_int = 1'bx;
      SIA_int = {2{1'bx}};
      SEA_int = 1'bx;
      DFTRAMBYP_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
  end

  always @ (CLKB_ or DFTRAMBYP_) begin
  	#0;
  	if(CLKB_ == 1'b1 && (DFTRAMBYP_int === 1'b1 || CENB_int != 1'b1)) begin
  	  if (RET1N_ == 1'b1) begin
  		SOB_int = ({DB_int_sh[18], DB_int_sh[0]});
  		DB_int_sh_int = DB_int_sh;
  	  end
  	end
  end
  always @ (SIA_int) begin
  	#0;
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1 && ^SIA_int === 1'bx) begin
	QA_int[9] = SIA_int[1]; 
	QA_int[8] = SIA_int[0]; 
  	end
  end

  always @ CLKA_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKA_ === 1'bx || CLKA_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
        QA_int = {19{1'bx}};
    end else if (CLKA_ === 1'b1 && LAST_CLKA === 1'b0) begin
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      SIA_int = SIA_;
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
  end else begin
      SIA_int = SIA_;
      SEA_int = SEA_;
      DFTRAMBYP_int = DFTRAMBYP_;
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      SIA_int = SIA_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
  if (RET1N_ == 1'b1) begin
	QA_int[18:9] = {QA_int[17:9], SIA_[1]}; 
	QA_int[8:0] = {SIA_[0], QA_int[8:1]}; 
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b0) begin
  if (RET1N_ == 1'b1) begin
	QA_int[18:9] = {QA_int[17:9], 1'b0}; 
	QA_int[8:0] = {1'b0, QA_int[8:1]}; 
    if (^{(CENA_int & !isBit1(DFTRAMBYP_int)), EMAA_int, EMASA_int, RET1N_int} === 1'bx) 
    ReadA;
  end
      end else begin
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      RET1N_int = RET1N_;
      SIA_int = SIA_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        AA_int = TENA_ ? AA_ : TAA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk0_int = 1'b0;
    ReadA;
      if (CENA_int === 1'b0) previous_CLKA = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int, 1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
      end
  end
    end else if (CLKA_ === 1'b0 && LAST_CLKA === 1'b1) begin
    end
  end
    LAST_CLKA = CLKA_;
  end

  reg globalNotifier0;
  initial globalNotifier0 = 1'b0;
  initial cont_flag0_int = 1'b0;

  always @ globalNotifier0 begin
    if ($realtime == 0) begin
    end else if ((EMAA_int[0] === 1'bx & DFTRAMBYP_int === 1'b1) || (EMAA_int[1] === 1'bx & DFTRAMBYP_int === 1'b1) || 
      (EMAA_int[2] === 1'bx & DFTRAMBYP_int === 1'b1) || (EMASA_int === 1'bx & DFTRAMBYP_int === 1'b1)
      ) begin
        QA_int = {19{1'bx}};
    end else if ((CENA_int === 1'bx & DFTRAMBYP_int === 1'b0) || EMAA_int[0] === 1'bx || 
      EMAA_int[1] === 1'bx || EMAA_int[2] === 1'bx || EMASA_int === 1'bx || RET1N_int === 1'bx
       || clk0_int === 1'bx) begin
        QA_int = {19{1'bx}};
    end else if (TENA_int === 1'bx) begin
      if(((CENA_ === 1'b1 & TCENA_ === 1'b1) & DFTRAMBYP_int === 1'b0) | (DFTRAMBYP_int === 1'b1 & SEA_int === 1'b1)) begin
      end else begin
      if (DFTRAMBYP_int === 1'b0) begin
        QA_int = {19{1'bx}};
      end
      end
    end else if (^SIA_int === 1'bx && DFTRAMBYP_int === 1'b1) begin
    end else if  (cont_flag0_int === 1'bx && COLLDISN_int === 1'b1 &&  (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
      AB_int, 1'b1, 1'b0)) begin
      cont_flag0_int = 1'b0;
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
    end else if  ((CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && cont_flag0_int === 1'bx && (COLLDISN_int === 1'b0 || COLLDISN_int === 
     1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
      cont_flag0_int = 1'b0;
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
    end else begin
      #0;
      ReadA;
   end
    globalNotifier0 = 1'b0;
  end
  always @ (CENB_ or TCENB_ or TENB_ or DFTRAMBYP_ or CLKB_) begin
  	if(CLKB_ == 1'b0) begin
  		CENB_p2 = CENB_;
  		TCENB_p2 = TCENB_;
  		DFTRAMBYP_p2 = DFTRAMBYP_;
  	end
  end

`ifdef POWER_PINS
  always @ (RET1N_ or VDDPE or VDDCE) begin
`else     
  always @ RET1N_ begin
`endif
`ifdef POWER_PINS
    if (RET1N_ == 1'b1 && RET1N_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st_b == 1'b1 && (CENB_ === 1'bx || TCENB_ === 1'bx || DFTRAMBYP_ === 1'bx || CLKB_ === 1'bx)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`endif
    if (RET1N_ === 1'bx || RET1N_ === 1'bz) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b0 && RET1N_int === 1'b1 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end else if (RET1N_ === 1'b1 && RET1N_int === 1'b0 && (CENB_p2 === 1'b0 || TCENB_p2 === 1'b0 || DFTRAMBYP_p2 === 1'b1)) begin
      failedWrite(1);
        QA_int = {19{1'bx}};
    end
`ifdef POWER_PINS
    if (RET1N_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st_b = 1;
      pre_charge_st = 1;
    end else if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(1);
      end
`else     
    if (RET1N_ == 1'b0) begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
      DB_int_sh = {19{1'bx}};
      DB_int_sh_int = {19{1'bx}};
      SOB_int = {2{1'bx}};
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SIB_int = {2{1'bx}};
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
`ifdef POWER_PINS
    end else if (RET1N_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st_b == 1'b1) begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
    end else begin
      pre_charge_st_b = 0;
      pre_charge_st = 0;
`else     
    end else begin
`endif
      CENB_int = 1'bx;
      AB_int = {8{1'bx}};
      DB_int = {19{1'bx}};
      DB_int_sh = {19{1'bx}};
      DB_int_sh_int = {19{1'bx}};
      SOB_int = {2{1'bx}};
      EMAB_int = {3{1'bx}};
      TENB_int = 1'bx;
      TCENB_int = 1'bx;
      TAB_int = {8{1'bx}};
      TDB_int = {19{1'bx}};
      RET1N_int = 1'bx;
      SIB_int = {2{1'bx}};
      SEB_int = 1'bx;
      COLLDISN_int = 1'bx;
    end
    RET1N_int = RET1N_;
  end

  always @ (SIB_int) begin
  	#0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1 && ^SIB_int === 1'bx) begin
	DB_int_sh_int[9] = SIB_int[1]; 
	DB_int_sh_int[8] = SIB_int[0]; 
  	end
  end
  always @ CLKB_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (RET1N_ == 1'b0 && VDDPE == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((CLKB_ === 1'bx || CLKB_ === 1'bz) && RET1N_ !== 1'b0) begin
      failedWrite(0);
    end else if (CLKB_ === 1'b1 && LAST_CLKB === 1'b0) begin
  if (RET1N_ == 1'b0) begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SIB_int = SIB_;
      SEB_int = SEB_;
  end else begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SIB_int = SIB_;
      SEB_int = SEB_;
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      SIB_int = SIB_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        DB_int_sh = TENB_ ? DB_ : TDB_;
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1) begin
      	DFTRAMBYP_int = DFTRAMBYP_;
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
	DB_int_sh[18:9] = {DB_int_sh_int[17:9], SIB_[1]}; 
	DB_int_sh[8:0] = {SIB_[0], DB_int_sh_int[8:1]}; 
      end else begin
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      TENB_int = TENB_;
      RET1N_int = RET1N_;
      SIB_int = SIB_;
      COLLDISN_int = COLLDISN_;
      	DFTRAMBYP_int = DFTRAMBYP_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        DB_int_sh_int = DB_int_sh;
        DB_int_sh = TENB_ ? DB_ : TDB_;
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        DFTRAMBYP_int = DFTRAMBYP_;
      end
      clk1_int = 1'b0;
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b0) begin
    if (^{(CENB_int & !isBit1(DFTRAMBYP_int)), EMAB_int, RET1N_int} === 1'bx) 
      WriteB;
      end else begin
      WriteB;
      end
      if (CENB_int === 1'b0) previous_CLKB = $realtime;
    #0;
      if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else if (((previous_CLKA == previous_CLKB)) && COLLDISN_int === 1'b1 && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && row_contention(AA_int,
        AB_int, 1'b1, 1'b0)) begin
`ifdef ARM_MESSAGES
          $display("%s row contention: in %m at %0t",ASSERT_PREFIX, $time);
`endif
          ROW_CC = 1;
`ifdef ARM_MESSAGES
          $display("%s contention: write B succeeds, read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end else if (((previous_CLKA == previous_CLKB)) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int 
       === 1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
      end
    end
      end
  end
  end
    LAST_CLKB = CLKB_;
  end

  reg globalNotifier1;
  initial globalNotifier1 = 1'b0;
  initial cont_flag1_int = 1'b0;

  always @ globalNotifier1 begin
    if ($realtime == 0) begin
    end else if ((EMAB_int[0] === 1'bx & DFTRAMBYP_int === 1'b1) || (EMAB_int[1] === 1'bx & DFTRAMBYP_int === 1'b1) || 
      (EMAB_int[2] === 1'bx & DFTRAMBYP_int === 1'b1)) begin
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if ((CENB_int === 1'bx & DFTRAMBYP_int === 1'b0) || EMAB_int[0] === 1'bx || 
      EMAB_int[1] === 1'bx || EMAB_int[2] === 1'bx || RET1N_int === 1'bx || clk1_int === 1'bx) begin
      failedWrite(1);
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
    end else if (TENB_int === 1'bx) begin
      if(((CENB_ === 1'b1 & TCENB_ === 1'b1) & DFTRAMBYP_int === 1'b0) | (DFTRAMBYP_int === 1'b1 & SEB_int === 1'b1)) begin
      end else begin
      if (DFTRAMBYP_int === 1'b0) begin
          failedWrite(1);
      end
  		#0;
  		SOB_int = {2{1'bx}};
  		DB_int_sh_int = {19{1'bx}};
  		DB_int_sh = {19{1'bx}};
      end
    end else if (^SIB_int === 1'bx && DFTRAMBYP_int === 1'b1) begin
    end else if  (cont_flag1_int === 1'bx && COLLDISN_int === 1'b1 &&  (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && is_contention(AA_int,
      AB_int, 1'b1, 1'b0)) begin
      cont_flag1_int = 1'b0;
          $display("%s contention: write B succeeds, read A fails in %m at %0t",ASSERT_PREFIX, $time);
          ROW_CC = 1;
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
    end else if  ((CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && cont_flag1_int === 1'bx && (COLLDISN_int === 1'b0 || COLLDISN_int === 
     1'bx) && row_contention(AA_int, AB_int,1'b1, 1'b0)) begin
      cont_flag1_int = 1'b0;
          ROW_CC = 1;
          $display("%s contention: write B fails in %m at %0t",ASSERT_PREFIX, $time);
          READ_WRITE = 1;
        DB_int = {19{1'bx}};
        WriteB;
        if (col_contention(AA_int,AB_int)) begin
          $display("%s contention: read A fails in %m at %0t",ASSERT_PREFIX, $time);
          COL_CC = 1;
          READ_WRITE = 1;
        QA_int = {19{1'bx}};
      end else begin
`ifdef ARM_MESSAGES
          $display("%s contention: read A succeeds in %m at %0t",ASSERT_PREFIX, $time);
`endif
          READ_WRITE = 1;
      end
    end else begin
      #0;
      WriteB;
   end
    globalNotifier1 = 1'b0;
  end
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
 always @ (VDDCE or VDDPE or VSSE) begin
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
 end
`endif

  function row_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
    reg sameRow;
    reg sameMux;
    reg anyWrite;
  begin
    anyWrite = ((& wena) === 1'b1 && (& wenb) === 1'b1) ? 1'b0 : 1'b1;
    sameMux = (aa[0:0] == ab[0:0]) ? 1'b1 : 1'b0;
    if (aa[7:1] == ab[7:1]) begin
      sameRow = 1'b1;
    end else begin
      sameRow = 1'b0;
    end
    if (sameRow == 1'b1 && anyWrite == 1'b1)
      row_contention = 1'b1;
    else if (sameRow == 1'b1 && sameMux == 1'b1)
      row_contention = 1'b1;
    else
      row_contention = 1'b0;
  end
  endfunction

  function col_contention;
    input [7:0] aa;
    input [7:0] ab;
  begin
    if (aa[0:0] == ab[0:0])
      col_contention = 1'b1;
    else
      col_contention = 1'b0;
  end
  endfunction

  function is_contention;
    input [7:0] aa;
    input [7:0] ab;
    input  wena;
    input  wenb;
    reg result;
  begin
    if ((& wena) === 1'b1 && (& wenb) === 1'b1) begin
      result = 1'b0;
    end else if (aa == ab) begin
      result = 1'b1;
    end else begin
      result = 1'b0;
    end
    is_contention = result;
  end
  endfunction

   wire contA_flag = (CENA_int !== 1'b1 && ((TENB_ ? CENB_ : TCENB_) !== 1'b1)) && ((COLLDISN_int === 1'b1 && is_contention(TENB_ ? AB_ : TAB_, AA_int, 1'b0, 1'b1)) ||
              ((COLLDISN_int === 1'b0 || COLLDISN_int === 1'bx) && row_contention(TENB_ ? AB_ : TAB_, AA_int, 1'b0, 1'b1)));
   wire contB_flag = (CENB_int !== 1'b1 && ((TENA_ ? CENA_ : TCENA_) !== 1'b1)) && ((COLLDISN_int === 1'b1 && is_contention(TENA_ ? AA_ : TAA_, AB_int, 1'b1, 1'b0)) ||
              ((COLLDISN_int === 1'b0 || COLLDISN_int === 1'bx) && row_contention(TENA_ ? AA_ : TAA_, AB_int, 1'b1, 1'b0)));

  always @ NOT_CENA begin
    CENA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA7 begin
    AA_int[7] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA6 begin
    AA_int[6] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA5 begin
    AA_int[5] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA4 begin
    AA_int[4] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA3 begin
    AA_int[3] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA2 begin
    AA_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA1 begin
    AA_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_AA0 begin
    AA_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CENB begin
    CENB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB7 begin
    AB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB6 begin
    AB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB5 begin
    AB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB4 begin
    AB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB3 begin
    AB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB2 begin
    AB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB1 begin
    AB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_AB0 begin
    AB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB18 begin
    DB_int[18] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB17 begin
    DB_int[17] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB16 begin
    DB_int[16] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB15 begin
    DB_int[15] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB14 begin
    DB_int[14] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB13 begin
    DB_int[13] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB12 begin
    DB_int[12] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB11 begin
    DB_int[11] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB10 begin
    DB_int[10] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB9 begin
    DB_int[9] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB8 begin
    DB_int[8] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB7 begin
    DB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB6 begin
    DB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB5 begin
    DB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB4 begin
    DB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB3 begin
    DB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB2 begin
    DB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB1 begin
    DB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_DB0 begin
    DB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_EMAA2 begin
    EMAA_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMAA1 begin
    EMAA_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMAA0 begin
    EMAA_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMASA begin
    EMASA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_EMAB2 begin
    EMAB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_EMAB1 begin
    EMAB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_EMAB0 begin
    EMAB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TENA begin
    TENA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TCENA begin
    CENA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA7 begin
    AA_int[7] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA6 begin
    AA_int[6] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA5 begin
    AA_int[5] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA4 begin
    AA_int[4] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA3 begin
    AA_int[3] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA2 begin
    AA_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA1 begin
    AA_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TAA0 begin
    AA_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_TENB begin
    TENB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TCENB begin
    CENB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB7 begin
    AB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB6 begin
    AB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB5 begin
    AB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB4 begin
    AB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB3 begin
    AB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB2 begin
    AB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB1 begin
    AB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TAB0 begin
    AB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB18 begin
    DB_int[18] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB17 begin
    DB_int[17] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB16 begin
    DB_int[16] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB15 begin
    DB_int[15] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB14 begin
    DB_int[14] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB13 begin
    DB_int[13] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB12 begin
    DB_int[12] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB11 begin
    DB_int[11] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB10 begin
    DB_int[10] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB9 begin
    DB_int[9] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB8 begin
    DB_int[8] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB7 begin
    DB_int[7] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB6 begin
    DB_int[6] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB5 begin
    DB_int[5] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB4 begin
    DB_int[4] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB3 begin
    DB_int[3] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB2 begin
    DB_int[2] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB1 begin
    DB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_TDB0 begin
    DB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_SIA1 begin
    SIA_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_SIA0 begin
    SIA_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_SEA begin
    SEA_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_DFTRAMBYP_CLKA begin
    DFTRAMBYP_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_DFTRAMBYP_CLKB begin
    DFTRAMBYP_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_RET1N begin
    RET1N_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_SIB1 begin
    SIB_int[1] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_SIB0 begin
    SIB_int[0] = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_SEB begin
    SEB_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_COLLDISN begin
    COLLDISN_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end

  always @ NOT_CONTA begin
    cont_flag0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CLKA_PER begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CLKA_MINH begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CLKA_MINL begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_CONTB begin
    cont_flag1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_CLKB_PER begin
    clk1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_CLKB_MINH begin
    clk1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end
  always @ NOT_CLKB_MINL begin
    clk1_int = 1'bx;
    if ( globalNotifier1 === 1'b0 ) globalNotifier1 = 1'bx;
  end


  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq1;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq1;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq1;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq0;
  wire contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq0;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq1;
  wire RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq1;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq0;
  wire contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq1;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq0;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq1;
  wire RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp;
  wire RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp;
  wire RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp;

  wire RET1Neq1aTENAeq1, RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0;
  wire RET1Neq1aTENBeq1, RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0;
  wire RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, RET1Neq1aTENAeq0;
  wire RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0;
  wire RET1Neq1aTENBeq0, RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0;
  wire RET1Neq1aSEAeq1, RET1Neq1aSEBeq1, RET1Neq1, RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp;
  wire RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp;

  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&!EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&!EMAA[1]&&EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&!EMAA[2]&&EMAA[1]&&EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&!EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&!EMAA[1]&&EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&EMAA[1]&&!EMAA[0] && contA_flag;
  assign contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENA&&!CENA)||(!TENA&&!TCENA))&&EMAA[2]&&EMAA[1]&&EMAA[0] && contA_flag;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&!EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq0 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&EMAA[0]&&!EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&!EMAA[1]&&EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&!EMAA[2]&&EMAA[1]&&EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&!EMAA[1]&&EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&!EMAA[0]&&EMASA;
  assign RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq1 = 
  RET1N&&((((TENA&&!CENA)||(!TENA&&!TCENA))&&!DFTRAMBYP)||DFTRAMBYP)&&EMAA[2]&&EMAA[1]&&EMAA[0]&&EMASA;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&!EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&!EMAB[1]&&EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&!EMAB[2]&&EMAB[1]&&EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&!EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&!EMAB[1]&&EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq0 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&EMAB[1]&&!EMAB[0] && contB_flag;
  assign contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq1 = 
  RET1N&&!DFTRAMBYP&&((TENB&&!CENB)||(!TENB&&!TCENB))&&EMAB[2]&&EMAB[1]&&EMAB[0] && contB_flag;
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&!EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&!EMAB[1]&&EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&!EMAB[2]&&EMAB[1]&&EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&!EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&!EMAB[1]&&EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq0 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&EMAB[1]&&!EMAB[0];
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq1 = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP)&&EMAB[2]&&EMAB[1]&&EMAB[0];
  assign RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp = 
  RET1N&&TENB&&((DFTRAMBYP&&!SEB)||(!DFTRAMBYP&&!CENB));
  assign RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp = 
  RET1N&&(((TENA&&!CENA&&!DFTRAMBYP)||(!TENA&&!TCENA&&!DFTRAMBYP))||DFTRAMBYP);
  assign RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp = 
  RET1N&&(((TENB&&!CENB&&!DFTRAMBYP)||(!TENB&&!TCENB&&!DFTRAMBYP))||DFTRAMBYP);
  assign RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp = 
  RET1N&&!TENB&&((DFTRAMBYP&&!SEB)||(!TCENB&&!DFTRAMBYP));

  assign RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1 = RET1N&&TENA&&!CENA&&COLLDISN;
  assign RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0 = RET1N&&TENA&&!CENA&&!COLLDISN;
  assign RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1 = RET1N&&TENB&&!CENB&&COLLDISN;
  assign RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0 = RET1N&&TENB&&!CENB&&!COLLDISN;
  assign RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1 = RET1N&&!TENA&&!TCENA&&COLLDISN;
  assign RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0 = RET1N&&!TENA&&!TCENA&&!COLLDISN;
  assign RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1 = RET1N&&!TENB&&!TCENB&&COLLDISN;
  assign RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0 = RET1N&&!TENB&&!TCENB&&!COLLDISN;
  assign RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp = RET1N&&((TENA&&!CENA)||(!TENA&&!TCENA));
  assign RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp = RET1N&&((TENB&&!CENB)||(!TENB&&!TCENB));


  assign RET1Neq1aTENAeq1 = RET1N&&TENA;
  assign RET1Neq1aTENBeq1 = RET1N&&TENB;
  assign RET1Neq1aTENAeq0 = RET1N&&!TENA;
  assign RET1Neq1aTENBeq0 = RET1N&&!TENB;
  assign RET1Neq1aSEAeq1 = RET1N&&SEA;
  assign RET1Neq1aSEBeq1 = RET1N&&SEB;
  assign RET1Neq1 = RET1N;

  specify

    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (CENA +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TCENA +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENA == 1'b0 && CENA == 1'b1)
       (TENA +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENA == 1'b1 && CENA == 1'b0)
       (TENA -=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> CENYA) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[7] +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[6] +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[5] +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[4] +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[3] +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[2] +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[1] +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b1)
       (AA[0] +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[7] +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[6] +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[5] +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[4] +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[3] +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[2] +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[1] +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENA == 1'b0)
       (TAA[0] +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[7] == 1'b0 && AA[7] == 1'b1)
       (TENA +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[6] == 1'b0 && AA[6] == 1'b1)
       (TENA +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[5] == 1'b0 && AA[5] == 1'b1)
       (TENA +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[4] == 1'b0 && AA[4] == 1'b1)
       (TENA +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[3] == 1'b0 && AA[3] == 1'b1)
       (TENA +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[2] == 1'b0 && AA[2] == 1'b1)
       (TENA +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[1] == 1'b0 && AA[1] == 1'b1)
       (TENA +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[0] == 1'b0 && AA[0] == 1'b1)
       (TENA +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[7] == 1'b1 && AA[7] == 1'b0)
       (TENA -=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[6] == 1'b1 && AA[6] == 1'b0)
       (TENA -=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[5] == 1'b1 && AA[5] == 1'b0)
       (TENA -=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[4] == 1'b1 && AA[4] == 1'b0)
       (TENA -=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[3] == 1'b1 && AA[3] == 1'b0)
       (TENA -=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[2] == 1'b1 && AA[2] == 1'b0)
       (TENA -=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[1] == 1'b1 && AA[1] == 1'b0)
       (TENA -=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAA[0] == 1'b1 && AA[0] == 1'b0)
       (TENA -=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYA[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (CENB +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TCENB +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENB == 1'b0 && CENB == 1'b1)
       (TENB +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TCENB == 1'b1 && CENB == 1'b0)
       (TENB -=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> CENYB) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[7] +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[6] +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[5] +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[4] +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[3] +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[2] +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[1] +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b1)
       (AB[0] +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[7] +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[6] +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[5] +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[4] +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[3] +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[2] +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[1] +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TENB == 1'b0)
       (TAB[0] +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[7] == 1'b0 && AB[7] == 1'b1)
       (TENB +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[6] == 1'b0 && AB[6] == 1'b1)
       (TENB +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[5] == 1'b0 && AB[5] == 1'b1)
       (TENB +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[4] == 1'b0 && AB[4] == 1'b1)
       (TENB +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[3] == 1'b0 && AB[3] == 1'b1)
       (TENB +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[2] == 1'b0 && AB[2] == 1'b1)
       (TENB +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[1] == 1'b0 && AB[1] == 1'b1)
       (TENB +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[0] == 1'b0 && AB[0] == 1'b1)
       (TENB +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[7] == 1'b1 && AB[7] == 1'b0)
       (TENB -=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[6] == 1'b1 && AB[6] == 1'b0)
       (TENB -=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[5] == 1'b1 && AB[5] == 1'b0)
       (TENB -=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[4] == 1'b1 && AB[4] == 1'b0)
       (TENB -=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[3] == 1'b1 && AB[3] == 1'b0)
       (TENB -=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[2] == 1'b1 && AB[2] == 1'b0)
       (TENB -=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[1] == 1'b1 && AB[1] == 1'b0)
       (TENB -=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (DFTRAMBYP == 1'b1 && TAB[0] == 1'b1 && AB[0] == 1'b0)
       (TENB -=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[7]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[6]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[5]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[4]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[3]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[2]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[1]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (DFTRAMBYP +=> AYB[0]) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[18] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[17] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[16] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (QA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b0 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b0 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b0 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b0)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1 && DFTRAMBYP == 1'b1 && EMAA[2] == 1'b1 && EMAA[1] == 1'b1 && EMAA[0] == 1'b1)
       (posedge CLKA => (SOA[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (posedge CLKB => (SOB[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (RET1N == 1'b1)
       (posedge CLKB => (SOB[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);


   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $period(posedge CLKA, `ARM_MEM_PERIOD, NOT_CLKA_PER);
   `else
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq0, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq0aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq0aEMAA1eq1aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq0aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq0aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
       $period(posedge CLKA &&& RET1Neq1aopopopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaDFTRAMBYPeq0cpoDFTRAMBYPeq1cpaEMAA2eq1aEMAA1eq1aEMAA0eq1aEMASAeq1, `ARM_MEM_PERIOD, NOT_CLKA_PER);
   `endif

   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $period(posedge CLKB, `ARM_MEM_PERIOD, NOT_CLKB_PER);
   `else
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq0aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq0aEMAB1eq1aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq0aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq0, `ARM_MEM_PERIOD, NOT_CLKB_PER);
       $period(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cpaEMAB2eq1aEMAB1eq1aEMAB0eq1, `ARM_MEM_PERIOD, NOT_CLKB_PER);
   `endif


   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $width(posedge CLKA, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINH);
       $width(negedge CLKA, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINL);
   `else
       $width(posedge CLKA &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINH);
       $width(negedge CLKA &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKA_MINL);
   `endif

   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $width(posedge CLKB, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINH);
       $width(negedge CLKB, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINL);
   `else
       $width(posedge CLKB &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINH);
       $width(negedge CLKB &&& RET1Neq1, `ARM_MEM_WIDTH, 0, NOT_CLKB_MINL);
   `endif


    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq0aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq0aEMAA1eq1aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq0aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq0, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);
    $setuphold(posedge CLKB &&& contA_RET1Neq1aDFTRAMBYPeq0aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcpaEMAA2eq1aEMAA1eq1aEMAA0eq1, posedge CLKA, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTA);

    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq0aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq0aEMAB1eq1aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq0aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq0, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);
    $setuphold(posedge CLKA &&& contB_RET1Neq1aDFTRAMBYPeq0aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcpaEMAB2eq1aEMAB1eq1aEMAB0eq1, posedge CLKB, 
    `ARM_MEM_COLLISION, 0.000, NOT_CONTB);

    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1, posedge CENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1, negedge CENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, posedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, posedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq1, negedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq1aCENAeq0aCOLLDISNeq0, negedge AA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AA0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1, posedge CENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1, negedge CENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_CENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, posedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, posedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq1, negedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aCENBeq0aCOLLDISNeq0, negedge AB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_AB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, posedge DB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq1aopopDFTRAMBYPeq1aSEBeq0cpoopDFTRAMBYPeq0aCENBeq0cpcp, negedge DB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DB0);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMASA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMASA);
    $setuphold(posedge CLKA &&& RET1Neq1aopopopTENAeq1aCENAeq0aDFTRAMBYPeq0cpoopTENAeq0aTCENAeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMASA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMASA);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, posedge EMAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aopopopTENBeq1aCENBeq0aDFTRAMBYPeq0cpoopTENBeq0aTCENBeq0aDFTRAMBYPeq0cpcpoDFTRAMBYPeq1cp, negedge EMAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_EMAB0);
    $setuphold(posedge CLKA &&& RET1Neq1, posedge TENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENA);
    $setuphold(posedge CLKA &&& RET1Neq1, negedge TENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0, posedge TCENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0, negedge TCENA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENA);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, posedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, posedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq1, negedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA7);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA6);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA5);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA4);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA3);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA2);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA1);
    $setuphold(posedge CLKA &&& RET1Neq1aTENAeq0aTCENAeq0aCOLLDISNeq0, negedge TAA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAA0);
    $setuphold(posedge CLKB &&& RET1Neq1, posedge TENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENB);
    $setuphold(posedge CLKB &&& RET1Neq1, negedge TENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0, posedge TCENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0, negedge TCENB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TCENB);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, posedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, posedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq1, negedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aTCENBeq0aCOLLDISNeq0, negedge TAB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TAB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, posedge TDB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB0);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[18], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB18);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[17], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB17);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[16], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB16);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB15);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB14);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB13);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB12);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB11);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB10);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB9);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB8);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB7);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB6);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB5);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB4);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB3);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB2);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB1);
    $setuphold(posedge CLKB &&& RET1Neq1aTENBeq0aopopDFTRAMBYPeq1aSEBeq0cpoopTCENBeq0aDFTRAMBYPeq0cpcp, negedge TDB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_TDB0);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, posedge SIA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA1);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, posedge SIA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA0);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, negedge SIA[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA1);
    $setuphold(posedge CLKA &&& RET1Neq1aSEAeq1, negedge SIA[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIA0);
    $setuphold(posedge CLKA &&& RET1Neq1, posedge SEA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEA);
    $setuphold(posedge CLKA &&& RET1Neq1, negedge SEA, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEA);
    $setuphold(posedge CLKA &&& RET1Neq1, posedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKA);
    $setuphold(posedge CLKA &&& RET1Neq1, negedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKA);
    $setuphold(posedge CLKB &&& RET1Neq1, posedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKB);
    $setuphold(posedge CLKB &&& RET1Neq1, negedge DFTRAMBYP, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_DFTRAMBYP_CLKB);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, posedge SIB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB1);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, posedge SIB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB0);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, negedge SIB[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB1);
    $setuphold(posedge CLKB &&& RET1Neq1aSEBeq1, negedge SIB[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SIB0);
    $setuphold(posedge CLKB &&& RET1Neq1, posedge SEB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEB);
    $setuphold(posedge CLKB &&& RET1Neq1, negedge SEB, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_SEB);
    $setuphold(posedge CLKA &&& RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp, posedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(posedge CLKA &&& RET1Neq1aopopTENAeq1aCENAeq0cpoopTENAeq0aTCENAeq0cpcp, negedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(posedge CLKB &&& RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp, posedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(posedge CLKB &&& RET1Neq1aopopTENBeq1aCENBeq0cpoopTENBeq0aTCENBeq0cpcp, negedge COLLDISN, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_COLLDISN);
    $setuphold(negedge RET1N, negedge CENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge CENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, negedge CENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge CENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, negedge TCENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge TCENA, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, negedge TCENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, negedge TCENB, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge DFTRAMBYP, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge DFTRAMBYP, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENB, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENA, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENA, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENB, negedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENB, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge TCENA, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENB, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge CENA, posedge RET1N, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(negedge RET1N, posedge DFTRAMBYP, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
    $setuphold(posedge RET1N, posedge DFTRAMBYP, 0.000, `ARM_MEM_HOLD, NOT_RET1N);
  endspecify


endmodule
`endcelldefine
  `endif
`endif
`timescale 1ns/1ps
module rf2_256x19_wm0_error_injection (Q_out, Q_in, CLK, A, CEN, DFTRAMBYP, SE);
   output [18:0] Q_out;
   input [18:0] Q_in;
   input CLK;
   input [7:0] A;
   input CEN;
   input DFTRAMBYP;
   input SE;
   parameter LEFT_RED_COLUMN_FAULT = 2'd1;
   parameter RIGHT_RED_COLUMN_FAULT = 2'd2;
   parameter NO_RED_FAULT = 2'd0;
   reg [18:0] Q_out;
   reg entry_found;
   reg list_complete;
   reg [17:0] fault_table [127:0];
   reg [17:0] fault_entry;
initial
begin
   `ifdef DUT
      `define pre_pend_path TB.DUT_inst.CHIP
   `else
       `define pre_pend_path TB.CHIP
   `endif
   `ifdef ARM_NONREPAIRABLE_FAULT
      `pre_pend_path.SMARCHCHKBVCD_LVISION_MBISTPG_ASSEMBLY_UNDER_TEST_INST.MEM0_MEM_INST.u1.add_fault(8'd163,5'd15,2'd1,2'd0);
   `endif
end
   task add_fault;
   //This task injects fault in memory
      input [7:0] address;
      input [4:0] bitPlace;
      input [1:0] fault_type;
      input [1:0] red_fault;
 
      integer i;
      reg done;
   begin
      done = 1'b0;
      i = 0;
      while ((!done) && i < 127)
      begin
         fault_entry = fault_table[i];
         if (fault_entry[0] === 1'b0 || fault_entry[0] === 1'bx)
         begin
            fault_entry[0] = 1'b1;
            fault_entry[2:1] = red_fault;
            fault_entry[4:3] = fault_type;
            fault_entry[9:5] = bitPlace;
            fault_entry[17:10] = address;
            fault_table[i] = fault_entry;
            done = 1'b1;
         end
         i = i+1;
      end
   end
   endtask
//This task removes all fault entries injected by user
task remove_all_faults;
   integer i;
begin
   for (i = 0; i < 128; i=i+1)
   begin
      fault_entry = fault_table[i];
      fault_entry[0] = 1'b0;
      fault_table[i] = fault_entry;
   end
end
endtask
task bit_error;
// This task is used to inject error in memory and should be called
// only from current module.
//
// This task injects error depending upon fault type to particular bit
// of the output
   inout [18:0] q_int;
   input [1:0] fault_type;
   input [4:0] bitLoc;
begin
   if (fault_type === 2'd0)
      q_int[bitLoc] = 1'b0;
   else if (fault_type === 2'd1)
      q_int[bitLoc] = 1'b1;
   else
      q_int[bitLoc] = ~q_int[bitLoc];
end
endtask
task error_injection_on_output;
// This function goes through error injection table for every
// read cycle and corrupts Q output if fault for the particular
// address is present in fault table
//
// If fault is redundant column is detected, this task corrupts
// Q output in read cycle
//
// If fault is repaired using repair bus, this task does not
// courrpt Q output in read cycle
//
   output [18:0] Q_output;
   reg list_complete;
   integer i;
   reg [6:0] row_address;
   reg [0:0] column_address;
   reg [4:0] bitPlace;
   reg [1:0] fault_type;
   reg [1:0] red_fault;
   reg valid;
   reg [3:0] msb_bit_calc;
begin
   entry_found = 1'b0;
   list_complete = 1'b0;
   i = 0;
   Q_output = Q_in;
   while(!list_complete)
   begin
      fault_entry = fault_table[i];
      {row_address, column_address, bitPlace, fault_type, red_fault, valid} = fault_entry;
      i = i + 1;
      if (valid == 1'b1)
      begin
         if (red_fault === NO_RED_FAULT)
         begin
            if (row_address == A[7:1] && column_address == A[0:0])
            begin
               if (bitPlace < 9)
                  bit_error(Q_output,fault_type, bitPlace);
               else if (bitPlace >= 9 )
                  bit_error(Q_output,fault_type, bitPlace);
            end
         end
      end
      else
         list_complete = 1'b1;
      end
   end
   endtask
   always @ (Q_in or CLK or A or CEN)
   begin
   if (CEN === 1'b0 && DFTRAMBYP === 1'b0 && SE === 1'b0)
      error_injection_on_output(Q_out);
   else
      Q_out = Q_in;
   end
endmodule
