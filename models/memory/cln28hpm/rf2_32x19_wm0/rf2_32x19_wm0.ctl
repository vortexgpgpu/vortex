/* ctl_memcomp Version: 4.0.5-EAC3 */
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
//      CTL model for High Density Two Port Register File SVT MVT Compiler
//
//       Instance Name:              rf2_32x19_wm0
//       Words:                      32
//       Bits:                       19
//       Mux:                        2
//       Drive:                      6
//       Write Mask:                 Off
//       Write Thru:                 Off
//       Extra Margin Adjustment:    On
//       Redundant Columns:          2
//       Test Muxes                  On
//       Power Gating:               Off
//       Retention:                  On
//       Pipeline:                   Off
//       Read Disturb Test:	        Off
//       
//       Creation Date:  Mon Nov 11 11:59:52 2019
//       Version: 	r4p0
STIL 1.0 {
   CTL P2001.10;
   Design P2001.01;
}
Header {
   Title "CTL model for `rf2_32x19_wm0";
}
Signals {
   "CENYA" Out;
   "AYA[4]" Out;
   "AYA[3]" Out;
   "AYA[2]" Out;
   "AYA[1]" Out;
   "AYA[0]" Out;
   "CENYB" Out;
   "AYB[4]" Out;
   "AYB[3]" Out;
   "AYB[2]" Out;
   "AYB[1]" Out;
   "AYB[0]" Out;
   "QA[18]" Out;
   "QA[17]" Out;
   "QA[16]" Out;
   "QA[15]" Out;
   "QA[14]" Out;
   "QA[13]" Out;
   "QA[12]" Out;
   "QA[11]" Out;
   "QA[10]" Out;
   "QA[9]" Out;
   "QA[8]" Out;
   "QA[7]" Out;
   "QA[6]" Out;
   "QA[5]" Out;
   "QA[4]" Out;
   "QA[3]" Out;
   "QA[2]" Out;
   "QA[1]" Out;
   "QA[0]" Out;
   "SOA[1]" Out;
   "SOA[0]" Out;
   "SOB[1]" Out;
   "SOB[0]" Out;
   "CLKA" In;
   "CENA" In;
   "AA[4]" In;
   "AA[3]" In;
   "AA[2]" In;
   "AA[1]" In;
   "AA[0]" In;
   "CLKB" In;
   "CENB" In;
   "AB[4]" In;
   "AB[3]" In;
   "AB[2]" In;
   "AB[1]" In;
   "AB[0]" In;
   "DB[18]" In;
   "DB[17]" In;
   "DB[16]" In;
   "DB[15]" In;
   "DB[14]" In;
   "DB[13]" In;
   "DB[12]" In;
   "DB[11]" In;
   "DB[10]" In;
   "DB[9]" In;
   "DB[8]" In;
   "DB[7]" In;
   "DB[6]" In;
   "DB[5]" In;
   "DB[4]" In;
   "DB[3]" In;
   "DB[2]" In;
   "DB[1]" In;
   "DB[0]" In;
   "EMAA[2]" In;
   "EMAA[1]" In;
   "EMAA[0]" In;
   "EMASA" In;
   "EMAB[2]" In;
   "EMAB[1]" In;
   "EMAB[0]" In;
   "TENA" In;
   "TCENA" In;
   "TAA[4]" In;
   "TAA[3]" In;
   "TAA[2]" In;
   "TAA[1]" In;
   "TAA[0]" In;
   "TENB" In;
   "TCENB" In;
   "TAB[4]" In;
   "TAB[3]" In;
   "TAB[2]" In;
   "TAB[1]" In;
   "TAB[0]" In;
   "TDB[18]" In;
   "TDB[17]" In;
   "TDB[16]" In;
   "TDB[15]" In;
   "TDB[14]" In;
   "TDB[13]" In;
   "TDB[12]" In;
   "TDB[11]" In;
   "TDB[10]" In;
   "TDB[9]" In;
   "TDB[8]" In;
   "TDB[7]" In;
   "TDB[6]" In;
   "TDB[5]" In;
   "TDB[4]" In;
   "TDB[3]" In;
   "TDB[2]" In;
   "TDB[1]" In;
   "TDB[0]" In;
   "RET1N" In;
   "SIA[1]" In;
   "SIA[0]" In;
   "SEA" In;
   "DFTRAMBYP" In;
   "SIB[1]" In;
   "SIB[0]" In;
   "SEB" In;
   "COLLDISN" In;
}
SignalGroups {
   "all_inputs" = '"CLKA" + "CENA" + "AA[4]" + "AA[3]" + "AA[2]" + "AA[1]" + "AA[0]" + 
   "CLKB" + "CENB" + "AB[4]" + "AB[3]" + "AB[2]" + "AB[1]" + "AB[0]" + "DB[18]" + 
   "DB[17]" + "DB[16]" + "DB[15]" + "DB[14]" + "DB[13]" + "DB[12]" + "DB[11]" + "DB[10]" + 
   "DB[9]" + "DB[8]" + "DB[7]" + "DB[6]" + "DB[5]" + "DB[4]" + "DB[3]" + "DB[2]" + 
   "DB[1]" + "DB[0]" + "EMAA[2]" + "EMAA[1]" + "EMAA[0]" + "EMASA" + "EMAB[2]" + 
   "EMAB[1]" + "EMAB[0]" + "TENA" + "TCENA" + "TAA[4]" + "TAA[3]" + "TAA[2]" + "TAA[1]" + 
   "TAA[0]" + "TENB" + "TCENB" + "TAB[4]" + "TAB[3]" + "TAB[2]" + "TAB[1]" + "TAB[0]" + 
   "TDB[18]" + "TDB[17]" + "TDB[16]" + "TDB[15]" + "TDB[14]" + "TDB[13]" + "TDB[12]" + 
   "TDB[11]" + "TDB[10]" + "TDB[9]" + "TDB[8]" + "TDB[7]" + "TDB[6]" + "TDB[5]" + 
   "TDB[4]" + "TDB[3]" + "TDB[2]" + "TDB[1]" + "TDB[0]" + "RET1N" + "SIA[1]" + "SIA[0]" + 
   "SEA" + "DFTRAMBYP" + "SIB[1]" + "SIB[0]" + "SEB" + "COLLDISN"';
   "all_outputs" = '"CENYA" + "AYA[4]" + "AYA[3]" + "AYA[2]" + "AYA[1]" + "AYA[0]" + 
   "CENYB" + "AYB[4]" + "AYB[3]" + "AYB[2]" + "AYB[1]" + "AYB[0]" + "QA[18]" + "QA[17]" + 
   "QA[16]" + "QA[15]" + "QA[14]" + "QA[13]" + "QA[12]" + "QA[11]" + "QA[10]" + "QA[9]" + 
   "QA[8]" + "QA[7]" + "QA[6]" + "QA[5]" + "QA[4]" + "QA[3]" + "QA[2]" + "QA[1]" + 
   "QA[0]" + "SOA[1]" + "SOA[0]" + "SOB[1]" + "SOB[0]"';
   "all_ports" = '"all_inputs" + "all_outputs"';
   "_pi" = '"CLKA" + "CENA" + "AA[4]" + "AA[3]" + "AA[2]" + "AA[1]" + "AA[0]" + "CLKB" + 
   "CENB" + "AB[4]" + "AB[3]" + "AB[2]" + "AB[1]" + "AB[0]" + "DB[18]" + "DB[17]" + 
   "DB[16]" + "DB[15]" + "DB[14]" + "DB[13]" + "DB[12]" + "DB[11]" + "DB[10]" + "DB[9]" + 
   "DB[8]" + "DB[7]" + "DB[6]" + "DB[5]" + "DB[4]" + "DB[3]" + "DB[2]" + "DB[1]" + 
   "DB[0]" + "EMAA[2]" + "EMAA[1]" + "EMAA[0]" + "EMASA" + "EMAB[2]" + "EMAB[1]" + 
   "EMAB[0]" + "TENA" + "TCENA" + "TAA[4]" + "TAA[3]" + "TAA[2]" + "TAA[1]" + "TAA[0]" + 
   "TENB" + "TCENB" + "TAB[4]" + "TAB[3]" + "TAB[2]" + "TAB[1]" + "TAB[0]" + "TDB[18]" + 
   "TDB[17]" + "TDB[16]" + "TDB[15]" + "TDB[14]" + "TDB[13]" + "TDB[12]" + "TDB[11]" + 
   "TDB[10]" + "TDB[9]" + "TDB[8]" + "TDB[7]" + "TDB[6]" + "TDB[5]" + "TDB[4]" + 
   "TDB[3]" + "TDB[2]" + "TDB[1]" + "TDB[0]" + "RET1N" + "SIA[1]" + "SIA[0]" + "SEA" + 
   "DFTRAMBYP" + "SIB[1]" + "SIB[0]" + "SEB" + "COLLDISN"';
   "_po" = '"CENYA" + "AYA[4]" + "AYA[3]" + "AYA[2]" + "AYA[1]" + "AYA[0]" + "CENYB" + 
   "AYB[4]" + "AYB[3]" + "AYB[2]" + "AYB[1]" + "AYB[0]" + "QA[18]" + "QA[17]" + "QA[16]" + 
   "QA[15]" + "QA[14]" + "QA[13]" + "QA[12]" + "QA[11]" + "QA[10]" + "QA[9]" + "QA[8]" + 
   "QA[7]" + "QA[6]" + "QA[5]" + "QA[4]" + "QA[3]" + "QA[2]" + "QA[1]" + "QA[0]" + 
   "SOA[1]" + "SOA[0]" + "SOB[1]" + "SOB[0]"';
   "_si" = '"SIA[0]" + "SIA[1]" + "SIB[0]" + "SIB[1]"' {ScanIn; }
   "_so" = '"SOA[0]" + "SOA[1]" + "SOB[0]" + "SOB[1]"' {ScanOut; }
}
ScanStructures {
   ScanChain "chain_rf2_32x19_wm0_1" {
      ScanLength  9;
      ScanCells   "uDQA8" "uDQA7" "uDQA6" "uDQA5" "uDQA4" "uDQA3" "uDQA2" "uDQA1" "uDQA0" ;
      ScanIn  "SIA[0]";
      ScanOut  "SOA[0]";
      ScanEnable  "SEA";
      ScanMasterClock  "CLKA";
   }
   ScanChain "chain_rf2_32x19_wm0_2" {
      ScanLength  10;
      ScanCells  "uDQA9" "uDQA10" "uDQA11" "uDQA12" "uDQA13" "uDQA14" "uDQA15" "uDQA16" "uDQA17" "uDQA18"  ;
      ScanIn  "SIA[1]";
      ScanOut  "SOA[1]";
      ScanEnable  "SEA";
      ScanMasterClock  "CLKA";
   }
   ScanChain "chain_rf2_32x19_wm0_3" {
      ScanLength  9;
      ScanCells   "uDQB8" "uDQB7" "uDQB6" "uDQB5" "uDQB4" "uDQB3" "uDQB2" "uDQB1" "uDQB0" ;
      ScanIn  "SIB[0]";
      ScanOut  "SOB[0]";
      ScanEnable  "SEB";
      ScanMasterClock  "CLKB";
   }
   ScanChain "chain_rf2_32x19_wm0_4" {
      ScanLength  10;
      ScanCells  "uDQB9" "uDQB10" "uDQB11" "uDQB12" "uDQB13" "uDQB14" "uDQB15" "uDQB16" "uDQB17" "uDQB18"  ;
      ScanIn  "SIB[1]";
      ScanOut  "SOB[1]";
      ScanEnable  "SEB";
      ScanMasterClock  "CLKB";
   }
}
Timing {
   WaveformTable "_default_WFT_" {
      Period '100ns';
      Waveforms {
         "all_inputs" {
            01ZN { '0ns' D/U/Z/N; }
         }
         "all_outputs" {
            XHTL { '40ns' X/H/T/L; }
         }
         "CLKA" {
            P { '0ns' D; '45ns' U; '55ns' D; }
         }
         "CLKB" {
            P { '0ns' D; '45ns' U; '55ns' D; }
         }
      }
   }
}
Procedures {
   "capture" {
      W "_default_WFT_";
      V { "_pi" = #; "_po" = #; }
   }
   "capture_CLK" {
      W "_default_WFT_";
      V {"_pi" = #; "_po" = #;"CLKA" = P;"CLKB" = P; }
   }
   "load_unload" {
      W "_default_WFT_";
      V { "CLKA" = 0; "CLKB" = 0; "_si" = \r2 N; "_so" =\r2 X; "SEA" = 1; "SEB" = 1; "DFTRAMBYP" = 1; }
      Shift {
         V { "CLKA" = P; "CLKB" = P; "_si" = \r2 #; "_so" = \r2 #; }
      }
   }
}
MacroDefs {
   "test_setup" {
      W "_default_WFT_";
      C {"all_inputs" = \r60 N; "all_outputs" = \r34 X; }
      V { "CLKA" = P; "CLKB" = P; }
   }
}
Environment "rf2_32x19_wm0" {
   CTL {
   }
   CTL Internal_scan {
      TestMode InternalTest;
      Focus Top {
      }
      Internal {
         "SIA[0]" {
            CaptureClock "CLKA" {
               LeadingEdge;
            }
            DataType ScanDataIn {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SIA[1]" {
            CaptureClock "CLKA" {
               LeadingEdge;
            }
            DataType ScanDataIn {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SOA[0]" {
            LaunchClock "CLKA" {
               LeadingEdge;
            }
            DataType ScanDataOut {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SOA[1]" {
            LaunchClock "CLKA" {
               LeadingEdge;
            }
            DataType ScanDataOut {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SEA" {
            DataType ScanEnable {
               ActiveState ForceUp;
            }
         }
         "CLKA" {
            DataType ScanMasterClock MasterClock;
         }
         "SIB[0]" {
            CaptureClock "CLKB" {
               LeadingEdge;
            }
            DataType ScanDataIn {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SIB[1]" {
            CaptureClock "CLKB" {
               LeadingEdge;
            }
            DataType ScanDataIn {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SOB[0]" {
            LaunchClock "CLKB" {
               LeadingEdge;
            }
            DataType ScanDataOut {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SOB[1]" {
            LaunchClock "CLKB" {
               LeadingEdge;
            }
            DataType ScanDataOut {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SEB" {
            DataType ScanEnable {
               ActiveState ForceUp;
            }
         }
         "CLKB" {
            DataType ScanMasterClock MasterClock;
         }
      }
   }
}
Environment dftSpec {
   CTL {
   }
   CTL all_dft {
      TestMode ForInheritOnly;
   }
}
