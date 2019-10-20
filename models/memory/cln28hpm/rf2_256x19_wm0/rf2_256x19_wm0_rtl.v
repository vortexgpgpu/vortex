/* verilog_rtl_memcomp Version: 4.0.5-beta11 */
/* common_memcomp Version: 4.0.5.2-amci */
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
//       Repair Verilog RTL for High Density Two Port Register File SVT MVT Compiler
//
//       Instance Name:              rf2_256x19_wm0_rtl_top
//       Words:                      256
//       User Bits:                  19
//       Mux:                        2
//       Drive:                      6
//       Write Mask:                 Off
//       Extra Margin Adjustment:    On
//       Redundancy:                 off
//       Redundant Rows:             0
//       Redundant Columns:          2
//       Test Muxes                  On
//       Ser:                        none
//       Retention:                  on
//       Power Gating:               off
//
//       Creation Date:  Sun Oct 20 14:44:23 2019
//       Version:      r4p0
//
//       Verified
//
//       Known Bugs: None.
//
//       Known Work Arounds: N/A
//
`timescale 1ns/1ps

module rf2_256x19_wm0_rtl_top (
          CENYA, 
          AYA, 
          CENYB, 
          AYB, 
          QA, 
          SOA, 
          SOB, 
          CLKA, 
          CENA, 
          AA, 
          CLKB, 
          CENB, 
          AB, 
          DB, 
          EMAA, 
          EMASA, 
          EMAB, 
          TENA, 
          TCENA, 
          TAA, 
          TENB, 
          TCENB, 
          TAB, 
          TDB, 
          RET1N, 
          SIA, 
          SEA, 
          DFTRAMBYP, 
          SIB, 
          SEB, 
          COLLDISN
   );

   output                   CENYA;
   output [7:0]             AYA;
   output                   CENYB;
   output [7:0]             AYB;
   output [18:0]            QA;
   output [1:0]             SOA;
   output [1:0]             SOB;
   input                    CLKA;
   input                    CENA;
   input [7:0]              AA;
   input                    CLKB;
   input                    CENB;
   input [7:0]              AB;
   input [18:0]             DB;
   input [2:0]              EMAA;
   input                    EMASA;
   input [2:0]              EMAB;
   input                    TENA;
   input                    TCENA;
   input [7:0]              TAA;
   input                    TENB;
   input                    TCENB;
   input [7:0]              TAB;
   input [18:0]             TDB;
   input                    RET1N;
   input [1:0]              SIA;
   input                    SEA;
   input                    DFTRAMBYP;
   input [1:0]              SIB;
   input                    SEB;
   input                    COLLDISN;
   wire [18:0]             QOA;
   wire [18:0]             DIB;

   assign QA = QOA;
   assign DIB = DB;
   rf2_256x19_wm0_fr_top u0 (
         .CENYA(CENYA),
         .AYA(AYA),
         .CENYB(CENYB),
         .AYB(AYB),
         .QOA(QOA),
         .SOA(SOA),
         .SOB(SOB),
         .CLKA(CLKA),
         .CENA(CENA),
         .AA(AA),
         .CLKB(CLKB),
         .CENB(CENB),
         .AB(AB),
         .DIB(DIB),
         .EMAA(EMAA),
         .EMASA(EMASA),
         .EMAB(EMAB),
         .TENA(TENA),
         .TCENA(TCENA),
         .TAA(TAA),
         .TENB(TENB),
         .TCENB(TCENB),
         .TAB(TAB),
         .TDB(TDB),
         .RET1N(RET1N),
         .SIA(SIA),
         .SEA(SEA),
         .DFTRAMBYP(DFTRAMBYP),
         .SIB(SIB),
         .SEB(SEB),
         .COLLDISN(COLLDISN)
);

endmodule

module rf2_256x19_wm0_fr_top (
          CENYA, 
          AYA, 
          CENYB, 
          AYB, 
          QOA, 
          SOA, 
          SOB, 
          CLKA, 
          CENA, 
          AA, 
          CLKB, 
          CENB, 
          AB, 
          DIB, 
          EMAA, 
          EMASA, 
          EMAB, 
          TENA, 
          TCENA, 
          TAA, 
          TENB, 
          TCENB, 
          TAB, 
          TDB, 
          RET1N, 
          SIA, 
          SEA, 
          DFTRAMBYP, 
          SIB, 
          SEB, 
          COLLDISN
   );

   output                   CENYA;
   output [7:0]             AYA;
   output                   CENYB;
   output [7:0]             AYB;
   output [18:0]            QOA;
   output [1:0]             SOA;
   output [1:0]             SOB;
   input                    CLKA;
   input                    CENA;
   input [7:0]              AA;
   input                    CLKB;
   input                    CENB;
   input [7:0]              AB;
   input [18:0]             DIB;
   input [2:0]              EMAA;
   input                    EMASA;
   input [2:0]              EMAB;
   input                    TENA;
   input                    TCENA;
   input [7:0]              TAA;
   input                    TENB;
   input                    TCENB;
   input [7:0]              TAB;
   input [18:0]             TDB;
   input                    RET1N;
   input [1:0]              SIA;
   input                    SEA;
   input                    DFTRAMBYP;
   input [1:0]              SIB;
   input                    SEB;
   input                    COLLDISN;

   wire [18:0]    DB;
   wire [18:0]    QA;

   assign DB=DIB;
   assign QOA=QA;
   rf2_256x19_wm0 u0 (
         .CENYA(CENYA),
         .AYA(AYA),
         .CENYB(CENYB),
         .AYB(AYB),
         .QA(QA),
         .SOA(SOA),
         .SOB(SOB),
         .CLKA(CLKA),
         .CENA(CENA),
         .AA(AA),
         .CLKB(CLKB),
         .CENB(CENB),
         .AB(AB),
         .DB(DB),
         .EMAA(EMAA),
         .EMASA(EMASA),
         .EMAB(EMAB),
         .TENA(TENA),
         .TCENA(TCENA),
         .TAA(TAA),
         .TENB(TENB),
         .TCENB(TCENB),
         .TAB(TAB),
         .TDB(TDB),
         .RET1N(RET1N),
         .SIA(SIA),
         .SEA(SEA),
         .DFTRAMBYP(DFTRAMBYP),
         .SIB(SIB),
         .SEB(SEB),
         .COLLDISN(COLLDISN)
   );

endmodule // rf2_256x19_wm0_fr_top

