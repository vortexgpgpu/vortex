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
//       Instance Name:              rf2_256x128_wm1
//       Words:                      256
//       Bits:                       128
//       Mux:                        2
//       Drive:                      6
//       Write Mask:                 On
//       Write Thru:                 Off
//       Extra Margin Adjustment:    On
//       Redundant Columns:          2
//       Test Muxes                  On
//       Power Gating:               Off
//       Retention:                  On
//       Pipeline:                   Off
//       Read Disturb Test:	        Off
//       
//       Creation Date:  Sun Oct 20 14:35:26 2019
//       Version: 	r4p0
STIL 1.0 {
   CTL P2001.10;
   Design P2001.01;
}
Header {
   Title "CTL model for `rf2_256x128_wm1";
}
Signals {
   "CENYA" Out;
   "AYA[7]" Out;
   "AYA[6]" Out;
   "AYA[5]" Out;
   "AYA[4]" Out;
   "AYA[3]" Out;
   "AYA[2]" Out;
   "AYA[1]" Out;
   "AYA[0]" Out;
   "CENYB" Out;
   "WENYB[127]" Out;
   "WENYB[126]" Out;
   "WENYB[125]" Out;
   "WENYB[124]" Out;
   "WENYB[123]" Out;
   "WENYB[122]" Out;
   "WENYB[121]" Out;
   "WENYB[120]" Out;
   "WENYB[119]" Out;
   "WENYB[118]" Out;
   "WENYB[117]" Out;
   "WENYB[116]" Out;
   "WENYB[115]" Out;
   "WENYB[114]" Out;
   "WENYB[113]" Out;
   "WENYB[112]" Out;
   "WENYB[111]" Out;
   "WENYB[110]" Out;
   "WENYB[109]" Out;
   "WENYB[108]" Out;
   "WENYB[107]" Out;
   "WENYB[106]" Out;
   "WENYB[105]" Out;
   "WENYB[104]" Out;
   "WENYB[103]" Out;
   "WENYB[102]" Out;
   "WENYB[101]" Out;
   "WENYB[100]" Out;
   "WENYB[99]" Out;
   "WENYB[98]" Out;
   "WENYB[97]" Out;
   "WENYB[96]" Out;
   "WENYB[95]" Out;
   "WENYB[94]" Out;
   "WENYB[93]" Out;
   "WENYB[92]" Out;
   "WENYB[91]" Out;
   "WENYB[90]" Out;
   "WENYB[89]" Out;
   "WENYB[88]" Out;
   "WENYB[87]" Out;
   "WENYB[86]" Out;
   "WENYB[85]" Out;
   "WENYB[84]" Out;
   "WENYB[83]" Out;
   "WENYB[82]" Out;
   "WENYB[81]" Out;
   "WENYB[80]" Out;
   "WENYB[79]" Out;
   "WENYB[78]" Out;
   "WENYB[77]" Out;
   "WENYB[76]" Out;
   "WENYB[75]" Out;
   "WENYB[74]" Out;
   "WENYB[73]" Out;
   "WENYB[72]" Out;
   "WENYB[71]" Out;
   "WENYB[70]" Out;
   "WENYB[69]" Out;
   "WENYB[68]" Out;
   "WENYB[67]" Out;
   "WENYB[66]" Out;
   "WENYB[65]" Out;
   "WENYB[64]" Out;
   "WENYB[63]" Out;
   "WENYB[62]" Out;
   "WENYB[61]" Out;
   "WENYB[60]" Out;
   "WENYB[59]" Out;
   "WENYB[58]" Out;
   "WENYB[57]" Out;
   "WENYB[56]" Out;
   "WENYB[55]" Out;
   "WENYB[54]" Out;
   "WENYB[53]" Out;
   "WENYB[52]" Out;
   "WENYB[51]" Out;
   "WENYB[50]" Out;
   "WENYB[49]" Out;
   "WENYB[48]" Out;
   "WENYB[47]" Out;
   "WENYB[46]" Out;
   "WENYB[45]" Out;
   "WENYB[44]" Out;
   "WENYB[43]" Out;
   "WENYB[42]" Out;
   "WENYB[41]" Out;
   "WENYB[40]" Out;
   "WENYB[39]" Out;
   "WENYB[38]" Out;
   "WENYB[37]" Out;
   "WENYB[36]" Out;
   "WENYB[35]" Out;
   "WENYB[34]" Out;
   "WENYB[33]" Out;
   "WENYB[32]" Out;
   "WENYB[31]" Out;
   "WENYB[30]" Out;
   "WENYB[29]" Out;
   "WENYB[28]" Out;
   "WENYB[27]" Out;
   "WENYB[26]" Out;
   "WENYB[25]" Out;
   "WENYB[24]" Out;
   "WENYB[23]" Out;
   "WENYB[22]" Out;
   "WENYB[21]" Out;
   "WENYB[20]" Out;
   "WENYB[19]" Out;
   "WENYB[18]" Out;
   "WENYB[17]" Out;
   "WENYB[16]" Out;
   "WENYB[15]" Out;
   "WENYB[14]" Out;
   "WENYB[13]" Out;
   "WENYB[12]" Out;
   "WENYB[11]" Out;
   "WENYB[10]" Out;
   "WENYB[9]" Out;
   "WENYB[8]" Out;
   "WENYB[7]" Out;
   "WENYB[6]" Out;
   "WENYB[5]" Out;
   "WENYB[4]" Out;
   "WENYB[3]" Out;
   "WENYB[2]" Out;
   "WENYB[1]" Out;
   "WENYB[0]" Out;
   "AYB[7]" Out;
   "AYB[6]" Out;
   "AYB[5]" Out;
   "AYB[4]" Out;
   "AYB[3]" Out;
   "AYB[2]" Out;
   "AYB[1]" Out;
   "AYB[0]" Out;
   "QA[127]" Out;
   "QA[126]" Out;
   "QA[125]" Out;
   "QA[124]" Out;
   "QA[123]" Out;
   "QA[122]" Out;
   "QA[121]" Out;
   "QA[120]" Out;
   "QA[119]" Out;
   "QA[118]" Out;
   "QA[117]" Out;
   "QA[116]" Out;
   "QA[115]" Out;
   "QA[114]" Out;
   "QA[113]" Out;
   "QA[112]" Out;
   "QA[111]" Out;
   "QA[110]" Out;
   "QA[109]" Out;
   "QA[108]" Out;
   "QA[107]" Out;
   "QA[106]" Out;
   "QA[105]" Out;
   "QA[104]" Out;
   "QA[103]" Out;
   "QA[102]" Out;
   "QA[101]" Out;
   "QA[100]" Out;
   "QA[99]" Out;
   "QA[98]" Out;
   "QA[97]" Out;
   "QA[96]" Out;
   "QA[95]" Out;
   "QA[94]" Out;
   "QA[93]" Out;
   "QA[92]" Out;
   "QA[91]" Out;
   "QA[90]" Out;
   "QA[89]" Out;
   "QA[88]" Out;
   "QA[87]" Out;
   "QA[86]" Out;
   "QA[85]" Out;
   "QA[84]" Out;
   "QA[83]" Out;
   "QA[82]" Out;
   "QA[81]" Out;
   "QA[80]" Out;
   "QA[79]" Out;
   "QA[78]" Out;
   "QA[77]" Out;
   "QA[76]" Out;
   "QA[75]" Out;
   "QA[74]" Out;
   "QA[73]" Out;
   "QA[72]" Out;
   "QA[71]" Out;
   "QA[70]" Out;
   "QA[69]" Out;
   "QA[68]" Out;
   "QA[67]" Out;
   "QA[66]" Out;
   "QA[65]" Out;
   "QA[64]" Out;
   "QA[63]" Out;
   "QA[62]" Out;
   "QA[61]" Out;
   "QA[60]" Out;
   "QA[59]" Out;
   "QA[58]" Out;
   "QA[57]" Out;
   "QA[56]" Out;
   "QA[55]" Out;
   "QA[54]" Out;
   "QA[53]" Out;
   "QA[52]" Out;
   "QA[51]" Out;
   "QA[50]" Out;
   "QA[49]" Out;
   "QA[48]" Out;
   "QA[47]" Out;
   "QA[46]" Out;
   "QA[45]" Out;
   "QA[44]" Out;
   "QA[43]" Out;
   "QA[42]" Out;
   "QA[41]" Out;
   "QA[40]" Out;
   "QA[39]" Out;
   "QA[38]" Out;
   "QA[37]" Out;
   "QA[36]" Out;
   "QA[35]" Out;
   "QA[34]" Out;
   "QA[33]" Out;
   "QA[32]" Out;
   "QA[31]" Out;
   "QA[30]" Out;
   "QA[29]" Out;
   "QA[28]" Out;
   "QA[27]" Out;
   "QA[26]" Out;
   "QA[25]" Out;
   "QA[24]" Out;
   "QA[23]" Out;
   "QA[22]" Out;
   "QA[21]" Out;
   "QA[20]" Out;
   "QA[19]" Out;
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
   "AA[7]" In;
   "AA[6]" In;
   "AA[5]" In;
   "AA[4]" In;
   "AA[3]" In;
   "AA[2]" In;
   "AA[1]" In;
   "AA[0]" In;
   "CLKB" In;
   "CENB" In;
   "WENB[127]" In;
   "WENB[126]" In;
   "WENB[125]" In;
   "WENB[124]" In;
   "WENB[123]" In;
   "WENB[122]" In;
   "WENB[121]" In;
   "WENB[120]" In;
   "WENB[119]" In;
   "WENB[118]" In;
   "WENB[117]" In;
   "WENB[116]" In;
   "WENB[115]" In;
   "WENB[114]" In;
   "WENB[113]" In;
   "WENB[112]" In;
   "WENB[111]" In;
   "WENB[110]" In;
   "WENB[109]" In;
   "WENB[108]" In;
   "WENB[107]" In;
   "WENB[106]" In;
   "WENB[105]" In;
   "WENB[104]" In;
   "WENB[103]" In;
   "WENB[102]" In;
   "WENB[101]" In;
   "WENB[100]" In;
   "WENB[99]" In;
   "WENB[98]" In;
   "WENB[97]" In;
   "WENB[96]" In;
   "WENB[95]" In;
   "WENB[94]" In;
   "WENB[93]" In;
   "WENB[92]" In;
   "WENB[91]" In;
   "WENB[90]" In;
   "WENB[89]" In;
   "WENB[88]" In;
   "WENB[87]" In;
   "WENB[86]" In;
   "WENB[85]" In;
   "WENB[84]" In;
   "WENB[83]" In;
   "WENB[82]" In;
   "WENB[81]" In;
   "WENB[80]" In;
   "WENB[79]" In;
   "WENB[78]" In;
   "WENB[77]" In;
   "WENB[76]" In;
   "WENB[75]" In;
   "WENB[74]" In;
   "WENB[73]" In;
   "WENB[72]" In;
   "WENB[71]" In;
   "WENB[70]" In;
   "WENB[69]" In;
   "WENB[68]" In;
   "WENB[67]" In;
   "WENB[66]" In;
   "WENB[65]" In;
   "WENB[64]" In;
   "WENB[63]" In;
   "WENB[62]" In;
   "WENB[61]" In;
   "WENB[60]" In;
   "WENB[59]" In;
   "WENB[58]" In;
   "WENB[57]" In;
   "WENB[56]" In;
   "WENB[55]" In;
   "WENB[54]" In;
   "WENB[53]" In;
   "WENB[52]" In;
   "WENB[51]" In;
   "WENB[50]" In;
   "WENB[49]" In;
   "WENB[48]" In;
   "WENB[47]" In;
   "WENB[46]" In;
   "WENB[45]" In;
   "WENB[44]" In;
   "WENB[43]" In;
   "WENB[42]" In;
   "WENB[41]" In;
   "WENB[40]" In;
   "WENB[39]" In;
   "WENB[38]" In;
   "WENB[37]" In;
   "WENB[36]" In;
   "WENB[35]" In;
   "WENB[34]" In;
   "WENB[33]" In;
   "WENB[32]" In;
   "WENB[31]" In;
   "WENB[30]" In;
   "WENB[29]" In;
   "WENB[28]" In;
   "WENB[27]" In;
   "WENB[26]" In;
   "WENB[25]" In;
   "WENB[24]" In;
   "WENB[23]" In;
   "WENB[22]" In;
   "WENB[21]" In;
   "WENB[20]" In;
   "WENB[19]" In;
   "WENB[18]" In;
   "WENB[17]" In;
   "WENB[16]" In;
   "WENB[15]" In;
   "WENB[14]" In;
   "WENB[13]" In;
   "WENB[12]" In;
   "WENB[11]" In;
   "WENB[10]" In;
   "WENB[9]" In;
   "WENB[8]" In;
   "WENB[7]" In;
   "WENB[6]" In;
   "WENB[5]" In;
   "WENB[4]" In;
   "WENB[3]" In;
   "WENB[2]" In;
   "WENB[1]" In;
   "WENB[0]" In;
   "AB[7]" In;
   "AB[6]" In;
   "AB[5]" In;
   "AB[4]" In;
   "AB[3]" In;
   "AB[2]" In;
   "AB[1]" In;
   "AB[0]" In;
   "DB[127]" In;
   "DB[126]" In;
   "DB[125]" In;
   "DB[124]" In;
   "DB[123]" In;
   "DB[122]" In;
   "DB[121]" In;
   "DB[120]" In;
   "DB[119]" In;
   "DB[118]" In;
   "DB[117]" In;
   "DB[116]" In;
   "DB[115]" In;
   "DB[114]" In;
   "DB[113]" In;
   "DB[112]" In;
   "DB[111]" In;
   "DB[110]" In;
   "DB[109]" In;
   "DB[108]" In;
   "DB[107]" In;
   "DB[106]" In;
   "DB[105]" In;
   "DB[104]" In;
   "DB[103]" In;
   "DB[102]" In;
   "DB[101]" In;
   "DB[100]" In;
   "DB[99]" In;
   "DB[98]" In;
   "DB[97]" In;
   "DB[96]" In;
   "DB[95]" In;
   "DB[94]" In;
   "DB[93]" In;
   "DB[92]" In;
   "DB[91]" In;
   "DB[90]" In;
   "DB[89]" In;
   "DB[88]" In;
   "DB[87]" In;
   "DB[86]" In;
   "DB[85]" In;
   "DB[84]" In;
   "DB[83]" In;
   "DB[82]" In;
   "DB[81]" In;
   "DB[80]" In;
   "DB[79]" In;
   "DB[78]" In;
   "DB[77]" In;
   "DB[76]" In;
   "DB[75]" In;
   "DB[74]" In;
   "DB[73]" In;
   "DB[72]" In;
   "DB[71]" In;
   "DB[70]" In;
   "DB[69]" In;
   "DB[68]" In;
   "DB[67]" In;
   "DB[66]" In;
   "DB[65]" In;
   "DB[64]" In;
   "DB[63]" In;
   "DB[62]" In;
   "DB[61]" In;
   "DB[60]" In;
   "DB[59]" In;
   "DB[58]" In;
   "DB[57]" In;
   "DB[56]" In;
   "DB[55]" In;
   "DB[54]" In;
   "DB[53]" In;
   "DB[52]" In;
   "DB[51]" In;
   "DB[50]" In;
   "DB[49]" In;
   "DB[48]" In;
   "DB[47]" In;
   "DB[46]" In;
   "DB[45]" In;
   "DB[44]" In;
   "DB[43]" In;
   "DB[42]" In;
   "DB[41]" In;
   "DB[40]" In;
   "DB[39]" In;
   "DB[38]" In;
   "DB[37]" In;
   "DB[36]" In;
   "DB[35]" In;
   "DB[34]" In;
   "DB[33]" In;
   "DB[32]" In;
   "DB[31]" In;
   "DB[30]" In;
   "DB[29]" In;
   "DB[28]" In;
   "DB[27]" In;
   "DB[26]" In;
   "DB[25]" In;
   "DB[24]" In;
   "DB[23]" In;
   "DB[22]" In;
   "DB[21]" In;
   "DB[20]" In;
   "DB[19]" In;
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
   "TAA[7]" In;
   "TAA[6]" In;
   "TAA[5]" In;
   "TAA[4]" In;
   "TAA[3]" In;
   "TAA[2]" In;
   "TAA[1]" In;
   "TAA[0]" In;
   "TENB" In;
   "TCENB" In;
   "TWENB[127]" In;
   "TWENB[126]" In;
   "TWENB[125]" In;
   "TWENB[124]" In;
   "TWENB[123]" In;
   "TWENB[122]" In;
   "TWENB[121]" In;
   "TWENB[120]" In;
   "TWENB[119]" In;
   "TWENB[118]" In;
   "TWENB[117]" In;
   "TWENB[116]" In;
   "TWENB[115]" In;
   "TWENB[114]" In;
   "TWENB[113]" In;
   "TWENB[112]" In;
   "TWENB[111]" In;
   "TWENB[110]" In;
   "TWENB[109]" In;
   "TWENB[108]" In;
   "TWENB[107]" In;
   "TWENB[106]" In;
   "TWENB[105]" In;
   "TWENB[104]" In;
   "TWENB[103]" In;
   "TWENB[102]" In;
   "TWENB[101]" In;
   "TWENB[100]" In;
   "TWENB[99]" In;
   "TWENB[98]" In;
   "TWENB[97]" In;
   "TWENB[96]" In;
   "TWENB[95]" In;
   "TWENB[94]" In;
   "TWENB[93]" In;
   "TWENB[92]" In;
   "TWENB[91]" In;
   "TWENB[90]" In;
   "TWENB[89]" In;
   "TWENB[88]" In;
   "TWENB[87]" In;
   "TWENB[86]" In;
   "TWENB[85]" In;
   "TWENB[84]" In;
   "TWENB[83]" In;
   "TWENB[82]" In;
   "TWENB[81]" In;
   "TWENB[80]" In;
   "TWENB[79]" In;
   "TWENB[78]" In;
   "TWENB[77]" In;
   "TWENB[76]" In;
   "TWENB[75]" In;
   "TWENB[74]" In;
   "TWENB[73]" In;
   "TWENB[72]" In;
   "TWENB[71]" In;
   "TWENB[70]" In;
   "TWENB[69]" In;
   "TWENB[68]" In;
   "TWENB[67]" In;
   "TWENB[66]" In;
   "TWENB[65]" In;
   "TWENB[64]" In;
   "TWENB[63]" In;
   "TWENB[62]" In;
   "TWENB[61]" In;
   "TWENB[60]" In;
   "TWENB[59]" In;
   "TWENB[58]" In;
   "TWENB[57]" In;
   "TWENB[56]" In;
   "TWENB[55]" In;
   "TWENB[54]" In;
   "TWENB[53]" In;
   "TWENB[52]" In;
   "TWENB[51]" In;
   "TWENB[50]" In;
   "TWENB[49]" In;
   "TWENB[48]" In;
   "TWENB[47]" In;
   "TWENB[46]" In;
   "TWENB[45]" In;
   "TWENB[44]" In;
   "TWENB[43]" In;
   "TWENB[42]" In;
   "TWENB[41]" In;
   "TWENB[40]" In;
   "TWENB[39]" In;
   "TWENB[38]" In;
   "TWENB[37]" In;
   "TWENB[36]" In;
   "TWENB[35]" In;
   "TWENB[34]" In;
   "TWENB[33]" In;
   "TWENB[32]" In;
   "TWENB[31]" In;
   "TWENB[30]" In;
   "TWENB[29]" In;
   "TWENB[28]" In;
   "TWENB[27]" In;
   "TWENB[26]" In;
   "TWENB[25]" In;
   "TWENB[24]" In;
   "TWENB[23]" In;
   "TWENB[22]" In;
   "TWENB[21]" In;
   "TWENB[20]" In;
   "TWENB[19]" In;
   "TWENB[18]" In;
   "TWENB[17]" In;
   "TWENB[16]" In;
   "TWENB[15]" In;
   "TWENB[14]" In;
   "TWENB[13]" In;
   "TWENB[12]" In;
   "TWENB[11]" In;
   "TWENB[10]" In;
   "TWENB[9]" In;
   "TWENB[8]" In;
   "TWENB[7]" In;
   "TWENB[6]" In;
   "TWENB[5]" In;
   "TWENB[4]" In;
   "TWENB[3]" In;
   "TWENB[2]" In;
   "TWENB[1]" In;
   "TWENB[0]" In;
   "TAB[7]" In;
   "TAB[6]" In;
   "TAB[5]" In;
   "TAB[4]" In;
   "TAB[3]" In;
   "TAB[2]" In;
   "TAB[1]" In;
   "TAB[0]" In;
   "TDB[127]" In;
   "TDB[126]" In;
   "TDB[125]" In;
   "TDB[124]" In;
   "TDB[123]" In;
   "TDB[122]" In;
   "TDB[121]" In;
   "TDB[120]" In;
   "TDB[119]" In;
   "TDB[118]" In;
   "TDB[117]" In;
   "TDB[116]" In;
   "TDB[115]" In;
   "TDB[114]" In;
   "TDB[113]" In;
   "TDB[112]" In;
   "TDB[111]" In;
   "TDB[110]" In;
   "TDB[109]" In;
   "TDB[108]" In;
   "TDB[107]" In;
   "TDB[106]" In;
   "TDB[105]" In;
   "TDB[104]" In;
   "TDB[103]" In;
   "TDB[102]" In;
   "TDB[101]" In;
   "TDB[100]" In;
   "TDB[99]" In;
   "TDB[98]" In;
   "TDB[97]" In;
   "TDB[96]" In;
   "TDB[95]" In;
   "TDB[94]" In;
   "TDB[93]" In;
   "TDB[92]" In;
   "TDB[91]" In;
   "TDB[90]" In;
   "TDB[89]" In;
   "TDB[88]" In;
   "TDB[87]" In;
   "TDB[86]" In;
   "TDB[85]" In;
   "TDB[84]" In;
   "TDB[83]" In;
   "TDB[82]" In;
   "TDB[81]" In;
   "TDB[80]" In;
   "TDB[79]" In;
   "TDB[78]" In;
   "TDB[77]" In;
   "TDB[76]" In;
   "TDB[75]" In;
   "TDB[74]" In;
   "TDB[73]" In;
   "TDB[72]" In;
   "TDB[71]" In;
   "TDB[70]" In;
   "TDB[69]" In;
   "TDB[68]" In;
   "TDB[67]" In;
   "TDB[66]" In;
   "TDB[65]" In;
   "TDB[64]" In;
   "TDB[63]" In;
   "TDB[62]" In;
   "TDB[61]" In;
   "TDB[60]" In;
   "TDB[59]" In;
   "TDB[58]" In;
   "TDB[57]" In;
   "TDB[56]" In;
   "TDB[55]" In;
   "TDB[54]" In;
   "TDB[53]" In;
   "TDB[52]" In;
   "TDB[51]" In;
   "TDB[50]" In;
   "TDB[49]" In;
   "TDB[48]" In;
   "TDB[47]" In;
   "TDB[46]" In;
   "TDB[45]" In;
   "TDB[44]" In;
   "TDB[43]" In;
   "TDB[42]" In;
   "TDB[41]" In;
   "TDB[40]" In;
   "TDB[39]" In;
   "TDB[38]" In;
   "TDB[37]" In;
   "TDB[36]" In;
   "TDB[35]" In;
   "TDB[34]" In;
   "TDB[33]" In;
   "TDB[32]" In;
   "TDB[31]" In;
   "TDB[30]" In;
   "TDB[29]" In;
   "TDB[28]" In;
   "TDB[27]" In;
   "TDB[26]" In;
   "TDB[25]" In;
   "TDB[24]" In;
   "TDB[23]" In;
   "TDB[22]" In;
   "TDB[21]" In;
   "TDB[20]" In;
   "TDB[19]" In;
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
   "all_inputs" = '"CLKA" + "CENA" + "AA[7]" + "AA[6]" + "AA[5]" + "AA[4]" + "AA[3]" + 
   "AA[2]" + "AA[1]" + "AA[0]" + "CLKB" + "CENB" + "WENB[127]" + "WENB[126]" + "WENB[125]" + 
   "WENB[124]" + "WENB[123]" + "WENB[122]" + "WENB[121]" + "WENB[120]" + "WENB[119]" + 
   "WENB[118]" + "WENB[117]" + "WENB[116]" + "WENB[115]" + "WENB[114]" + "WENB[113]" + 
   "WENB[112]" + "WENB[111]" + "WENB[110]" + "WENB[109]" + "WENB[108]" + "WENB[107]" + 
   "WENB[106]" + "WENB[105]" + "WENB[104]" + "WENB[103]" + "WENB[102]" + "WENB[101]" + 
   "WENB[100]" + "WENB[99]" + "WENB[98]" + "WENB[97]" + "WENB[96]" + "WENB[95]" + 
   "WENB[94]" + "WENB[93]" + "WENB[92]" + "WENB[91]" + "WENB[90]" + "WENB[89]" + 
   "WENB[88]" + "WENB[87]" + "WENB[86]" + "WENB[85]" + "WENB[84]" + "WENB[83]" + 
   "WENB[82]" + "WENB[81]" + "WENB[80]" + "WENB[79]" + "WENB[78]" + "WENB[77]" + 
   "WENB[76]" + "WENB[75]" + "WENB[74]" + "WENB[73]" + "WENB[72]" + "WENB[71]" + 
   "WENB[70]" + "WENB[69]" + "WENB[68]" + "WENB[67]" + "WENB[66]" + "WENB[65]" + 
   "WENB[64]" + "WENB[63]" + "WENB[62]" + "WENB[61]" + "WENB[60]" + "WENB[59]" + 
   "WENB[58]" + "WENB[57]" + "WENB[56]" + "WENB[55]" + "WENB[54]" + "WENB[53]" + 
   "WENB[52]" + "WENB[51]" + "WENB[50]" + "WENB[49]" + "WENB[48]" + "WENB[47]" + 
   "WENB[46]" + "WENB[45]" + "WENB[44]" + "WENB[43]" + "WENB[42]" + "WENB[41]" + 
   "WENB[40]" + "WENB[39]" + "WENB[38]" + "WENB[37]" + "WENB[36]" + "WENB[35]" + 
   "WENB[34]" + "WENB[33]" + "WENB[32]" + "WENB[31]" + "WENB[30]" + "WENB[29]" + 
   "WENB[28]" + "WENB[27]" + "WENB[26]" + "WENB[25]" + "WENB[24]" + "WENB[23]" + 
   "WENB[22]" + "WENB[21]" + "WENB[20]" + "WENB[19]" + "WENB[18]" + "WENB[17]" + 
   "WENB[16]" + "WENB[15]" + "WENB[14]" + "WENB[13]" + "WENB[12]" + "WENB[11]" + 
   "WENB[10]" + "WENB[9]" + "WENB[8]" + "WENB[7]" + "WENB[6]" + "WENB[5]" + "WENB[4]" + 
   "WENB[3]" + "WENB[2]" + "WENB[1]" + "WENB[0]" + "AB[7]" + "AB[6]" + "AB[5]" + 
   "AB[4]" + "AB[3]" + "AB[2]" + "AB[1]" + "AB[0]" + "DB[127]" + "DB[126]" + "DB[125]" + 
   "DB[124]" + "DB[123]" + "DB[122]" + "DB[121]" + "DB[120]" + "DB[119]" + "DB[118]" + 
   "DB[117]" + "DB[116]" + "DB[115]" + "DB[114]" + "DB[113]" + "DB[112]" + "DB[111]" + 
   "DB[110]" + "DB[109]" + "DB[108]" + "DB[107]" + "DB[106]" + "DB[105]" + "DB[104]" + 
   "DB[103]" + "DB[102]" + "DB[101]" + "DB[100]" + "DB[99]" + "DB[98]" + "DB[97]" + 
   "DB[96]" + "DB[95]" + "DB[94]" + "DB[93]" + "DB[92]" + "DB[91]" + "DB[90]" + "DB[89]" + 
   "DB[88]" + "DB[87]" + "DB[86]" + "DB[85]" + "DB[84]" + "DB[83]" + "DB[82]" + "DB[81]" + 
   "DB[80]" + "DB[79]" + "DB[78]" + "DB[77]" + "DB[76]" + "DB[75]" + "DB[74]" + "DB[73]" + 
   "DB[72]" + "DB[71]" + "DB[70]" + "DB[69]" + "DB[68]" + "DB[67]" + "DB[66]" + "DB[65]" + 
   "DB[64]" + "DB[63]" + "DB[62]" + "DB[61]" + "DB[60]" + "DB[59]" + "DB[58]" + "DB[57]" + 
   "DB[56]" + "DB[55]" + "DB[54]" + "DB[53]" + "DB[52]" + "DB[51]" + "DB[50]" + "DB[49]" + 
   "DB[48]" + "DB[47]" + "DB[46]" + "DB[45]" + "DB[44]" + "DB[43]" + "DB[42]" + "DB[41]" + 
   "DB[40]" + "DB[39]" + "DB[38]" + "DB[37]" + "DB[36]" + "DB[35]" + "DB[34]" + "DB[33]" + 
   "DB[32]" + "DB[31]" + "DB[30]" + "DB[29]" + "DB[28]" + "DB[27]" + "DB[26]" + "DB[25]" + 
   "DB[24]" + "DB[23]" + "DB[22]" + "DB[21]" + "DB[20]" + "DB[19]" + "DB[18]" + "DB[17]" + 
   "DB[16]" + "DB[15]" + "DB[14]" + "DB[13]" + "DB[12]" + "DB[11]" + "DB[10]" + "DB[9]" + 
   "DB[8]" + "DB[7]" + "DB[6]" + "DB[5]" + "DB[4]" + "DB[3]" + "DB[2]" + "DB[1]" + 
   "DB[0]" + "EMAA[2]" + "EMAA[1]" + "EMAA[0]" + "EMASA" + "EMAB[2]" + "EMAB[1]" + 
   "EMAB[0]" + "TENA" + "TCENA" + "TAA[7]" + "TAA[6]" + "TAA[5]" + "TAA[4]" + "TAA[3]" + 
   "TAA[2]" + "TAA[1]" + "TAA[0]" + "TENB" + "TCENB" + "TWENB[127]" + "TWENB[126]" + 
   "TWENB[125]" + "TWENB[124]" + "TWENB[123]" + "TWENB[122]" + "TWENB[121]" + "TWENB[120]" + 
   "TWENB[119]" + "TWENB[118]" + "TWENB[117]" + "TWENB[116]" + "TWENB[115]" + "TWENB[114]" + 
   "TWENB[113]" + "TWENB[112]" + "TWENB[111]" + "TWENB[110]" + "TWENB[109]" + "TWENB[108]" + 
   "TWENB[107]" + "TWENB[106]" + "TWENB[105]" + "TWENB[104]" + "TWENB[103]" + "TWENB[102]" + 
   "TWENB[101]" + "TWENB[100]" + "TWENB[99]" + "TWENB[98]" + "TWENB[97]" + "TWENB[96]" + 
   "TWENB[95]" + "TWENB[94]" + "TWENB[93]" + "TWENB[92]" + "TWENB[91]" + "TWENB[90]" + 
   "TWENB[89]" + "TWENB[88]" + "TWENB[87]" + "TWENB[86]" + "TWENB[85]" + "TWENB[84]" + 
   "TWENB[83]" + "TWENB[82]" + "TWENB[81]" + "TWENB[80]" + "TWENB[79]" + "TWENB[78]" + 
   "TWENB[77]" + "TWENB[76]" + "TWENB[75]" + "TWENB[74]" + "TWENB[73]" + "TWENB[72]" + 
   "TWENB[71]" + "TWENB[70]" + "TWENB[69]" + "TWENB[68]" + "TWENB[67]" + "TWENB[66]" + 
   "TWENB[65]" + "TWENB[64]" + "TWENB[63]" + "TWENB[62]" + "TWENB[61]" + "TWENB[60]" + 
   "TWENB[59]" + "TWENB[58]" + "TWENB[57]" + "TWENB[56]" + "TWENB[55]" + "TWENB[54]" + 
   "TWENB[53]" + "TWENB[52]" + "TWENB[51]" + "TWENB[50]" + "TWENB[49]" + "TWENB[48]" + 
   "TWENB[47]" + "TWENB[46]" + "TWENB[45]" + "TWENB[44]" + "TWENB[43]" + "TWENB[42]" + 
   "TWENB[41]" + "TWENB[40]" + "TWENB[39]" + "TWENB[38]" + "TWENB[37]" + "TWENB[36]" + 
   "TWENB[35]" + "TWENB[34]" + "TWENB[33]" + "TWENB[32]" + "TWENB[31]" + "TWENB[30]" + 
   "TWENB[29]" + "TWENB[28]" + "TWENB[27]" + "TWENB[26]" + "TWENB[25]" + "TWENB[24]" + 
   "TWENB[23]" + "TWENB[22]" + "TWENB[21]" + "TWENB[20]" + "TWENB[19]" + "TWENB[18]" + 
   "TWENB[17]" + "TWENB[16]" + "TWENB[15]" + "TWENB[14]" + "TWENB[13]" + "TWENB[12]" + 
   "TWENB[11]" + "TWENB[10]" + "TWENB[9]" + "TWENB[8]" + "TWENB[7]" + "TWENB[6]" + 
   "TWENB[5]" + "TWENB[4]" + "TWENB[3]" + "TWENB[2]" + "TWENB[1]" + "TWENB[0]" + 
   "TAB[7]" + "TAB[6]" + "TAB[5]" + "TAB[4]" + "TAB[3]" + "TAB[2]" + "TAB[1]" + "TAB[0]" + 
   "TDB[127]" + "TDB[126]" + "TDB[125]" + "TDB[124]" + "TDB[123]" + "TDB[122]" + 
   "TDB[121]" + "TDB[120]" + "TDB[119]" + "TDB[118]" + "TDB[117]" + "TDB[116]" + 
   "TDB[115]" + "TDB[114]" + "TDB[113]" + "TDB[112]" + "TDB[111]" + "TDB[110]" + 
   "TDB[109]" + "TDB[108]" + "TDB[107]" + "TDB[106]" + "TDB[105]" + "TDB[104]" + 
   "TDB[103]" + "TDB[102]" + "TDB[101]" + "TDB[100]" + "TDB[99]" + "TDB[98]" + "TDB[97]" + 
   "TDB[96]" + "TDB[95]" + "TDB[94]" + "TDB[93]" + "TDB[92]" + "TDB[91]" + "TDB[90]" + 
   "TDB[89]" + "TDB[88]" + "TDB[87]" + "TDB[86]" + "TDB[85]" + "TDB[84]" + "TDB[83]" + 
   "TDB[82]" + "TDB[81]" + "TDB[80]" + "TDB[79]" + "TDB[78]" + "TDB[77]" + "TDB[76]" + 
   "TDB[75]" + "TDB[74]" + "TDB[73]" + "TDB[72]" + "TDB[71]" + "TDB[70]" + "TDB[69]" + 
   "TDB[68]" + "TDB[67]" + "TDB[66]" + "TDB[65]" + "TDB[64]" + "TDB[63]" + "TDB[62]" + 
   "TDB[61]" + "TDB[60]" + "TDB[59]" + "TDB[58]" + "TDB[57]" + "TDB[56]" + "TDB[55]" + 
   "TDB[54]" + "TDB[53]" + "TDB[52]" + "TDB[51]" + "TDB[50]" + "TDB[49]" + "TDB[48]" + 
   "TDB[47]" + "TDB[46]" + "TDB[45]" + "TDB[44]" + "TDB[43]" + "TDB[42]" + "TDB[41]" + 
   "TDB[40]" + "TDB[39]" + "TDB[38]" + "TDB[37]" + "TDB[36]" + "TDB[35]" + "TDB[34]" + 
   "TDB[33]" + "TDB[32]" + "TDB[31]" + "TDB[30]" + "TDB[29]" + "TDB[28]" + "TDB[27]" + 
   "TDB[26]" + "TDB[25]" + "TDB[24]" + "TDB[23]" + "TDB[22]" + "TDB[21]" + "TDB[20]" + 
   "TDB[19]" + "TDB[18]" + "TDB[17]" + "TDB[16]" + "TDB[15]" + "TDB[14]" + "TDB[13]" + 
   "TDB[12]" + "TDB[11]" + "TDB[10]" + "TDB[9]" + "TDB[8]" + "TDB[7]" + "TDB[6]" + 
   "TDB[5]" + "TDB[4]" + "TDB[3]" + "TDB[2]" + "TDB[1]" + "TDB[0]" + "RET1N" + "SIA[1]" + 
   "SIA[0]" + "SEA" + "DFTRAMBYP" + "SIB[1]" + "SIB[0]" + "SEB" + "COLLDISN"';
   "all_outputs" = '"CENYA" + "AYA[7]" + "AYA[6]" + "AYA[5]" + "AYA[4]" + "AYA[3]" + 
   "AYA[2]" + "AYA[1]" + "AYA[0]" + "CENYB" + "WENYB[127]" + "WENYB[126]" + "WENYB[125]" + 
   "WENYB[124]" + "WENYB[123]" + "WENYB[122]" + "WENYB[121]" + "WENYB[120]" + "WENYB[119]" + 
   "WENYB[118]" + "WENYB[117]" + "WENYB[116]" + "WENYB[115]" + "WENYB[114]" + "WENYB[113]" + 
   "WENYB[112]" + "WENYB[111]" + "WENYB[110]" + "WENYB[109]" + "WENYB[108]" + "WENYB[107]" + 
   "WENYB[106]" + "WENYB[105]" + "WENYB[104]" + "WENYB[103]" + "WENYB[102]" + "WENYB[101]" + 
   "WENYB[100]" + "WENYB[99]" + "WENYB[98]" + "WENYB[97]" + "WENYB[96]" + "WENYB[95]" + 
   "WENYB[94]" + "WENYB[93]" + "WENYB[92]" + "WENYB[91]" + "WENYB[90]" + "WENYB[89]" + 
   "WENYB[88]" + "WENYB[87]" + "WENYB[86]" + "WENYB[85]" + "WENYB[84]" + "WENYB[83]" + 
   "WENYB[82]" + "WENYB[81]" + "WENYB[80]" + "WENYB[79]" + "WENYB[78]" + "WENYB[77]" + 
   "WENYB[76]" + "WENYB[75]" + "WENYB[74]" + "WENYB[73]" + "WENYB[72]" + "WENYB[71]" + 
   "WENYB[70]" + "WENYB[69]" + "WENYB[68]" + "WENYB[67]" + "WENYB[66]" + "WENYB[65]" + 
   "WENYB[64]" + "WENYB[63]" + "WENYB[62]" + "WENYB[61]" + "WENYB[60]" + "WENYB[59]" + 
   "WENYB[58]" + "WENYB[57]" + "WENYB[56]" + "WENYB[55]" + "WENYB[54]" + "WENYB[53]" + 
   "WENYB[52]" + "WENYB[51]" + "WENYB[50]" + "WENYB[49]" + "WENYB[48]" + "WENYB[47]" + 
   "WENYB[46]" + "WENYB[45]" + "WENYB[44]" + "WENYB[43]" + "WENYB[42]" + "WENYB[41]" + 
   "WENYB[40]" + "WENYB[39]" + "WENYB[38]" + "WENYB[37]" + "WENYB[36]" + "WENYB[35]" + 
   "WENYB[34]" + "WENYB[33]" + "WENYB[32]" + "WENYB[31]" + "WENYB[30]" + "WENYB[29]" + 
   "WENYB[28]" + "WENYB[27]" + "WENYB[26]" + "WENYB[25]" + "WENYB[24]" + "WENYB[23]" + 
   "WENYB[22]" + "WENYB[21]" + "WENYB[20]" + "WENYB[19]" + "WENYB[18]" + "WENYB[17]" + 
   "WENYB[16]" + "WENYB[15]" + "WENYB[14]" + "WENYB[13]" + "WENYB[12]" + "WENYB[11]" + 
   "WENYB[10]" + "WENYB[9]" + "WENYB[8]" + "WENYB[7]" + "WENYB[6]" + "WENYB[5]" + 
   "WENYB[4]" + "WENYB[3]" + "WENYB[2]" + "WENYB[1]" + "WENYB[0]" + "AYB[7]" + "AYB[6]" + 
   "AYB[5]" + "AYB[4]" + "AYB[3]" + "AYB[2]" + "AYB[1]" + "AYB[0]" + "QA[127]" + 
   "QA[126]" + "QA[125]" + "QA[124]" + "QA[123]" + "QA[122]" + "QA[121]" + "QA[120]" + 
   "QA[119]" + "QA[118]" + "QA[117]" + "QA[116]" + "QA[115]" + "QA[114]" + "QA[113]" + 
   "QA[112]" + "QA[111]" + "QA[110]" + "QA[109]" + "QA[108]" + "QA[107]" + "QA[106]" + 
   "QA[105]" + "QA[104]" + "QA[103]" + "QA[102]" + "QA[101]" + "QA[100]" + "QA[99]" + 
   "QA[98]" + "QA[97]" + "QA[96]" + "QA[95]" + "QA[94]" + "QA[93]" + "QA[92]" + "QA[91]" + 
   "QA[90]" + "QA[89]" + "QA[88]" + "QA[87]" + "QA[86]" + "QA[85]" + "QA[84]" + "QA[83]" + 
   "QA[82]" + "QA[81]" + "QA[80]" + "QA[79]" + "QA[78]" + "QA[77]" + "QA[76]" + "QA[75]" + 
   "QA[74]" + "QA[73]" + "QA[72]" + "QA[71]" + "QA[70]" + "QA[69]" + "QA[68]" + "QA[67]" + 
   "QA[66]" + "QA[65]" + "QA[64]" + "QA[63]" + "QA[62]" + "QA[61]" + "QA[60]" + "QA[59]" + 
   "QA[58]" + "QA[57]" + "QA[56]" + "QA[55]" + "QA[54]" + "QA[53]" + "QA[52]" + "QA[51]" + 
   "QA[50]" + "QA[49]" + "QA[48]" + "QA[47]" + "QA[46]" + "QA[45]" + "QA[44]" + "QA[43]" + 
   "QA[42]" + "QA[41]" + "QA[40]" + "QA[39]" + "QA[38]" + "QA[37]" + "QA[36]" + "QA[35]" + 
   "QA[34]" + "QA[33]" + "QA[32]" + "QA[31]" + "QA[30]" + "QA[29]" + "QA[28]" + "QA[27]" + 
   "QA[26]" + "QA[25]" + "QA[24]" + "QA[23]" + "QA[22]" + "QA[21]" + "QA[20]" + "QA[19]" + 
   "QA[18]" + "QA[17]" + "QA[16]" + "QA[15]" + "QA[14]" + "QA[13]" + "QA[12]" + "QA[11]" + 
   "QA[10]" + "QA[9]" + "QA[8]" + "QA[7]" + "QA[6]" + "QA[5]" + "QA[4]" + "QA[3]" + 
   "QA[2]" + "QA[1]" + "QA[0]" + "SOA[1]" + "SOA[0]" + "SOB[1]" + "SOB[0]"';
   "all_ports" = '"all_inputs" + "all_outputs"';
   "_pi" = '"CLKA" + "CENA" + "AA[7]" + "AA[6]" + "AA[5]" + "AA[4]" + "AA[3]" + "AA[2]" + 
   "AA[1]" + "AA[0]" + "CLKB" + "CENB" + "WENB[127]" + "WENB[126]" + "WENB[125]" + 
   "WENB[124]" + "WENB[123]" + "WENB[122]" + "WENB[121]" + "WENB[120]" + "WENB[119]" + 
   "WENB[118]" + "WENB[117]" + "WENB[116]" + "WENB[115]" + "WENB[114]" + "WENB[113]" + 
   "WENB[112]" + "WENB[111]" + "WENB[110]" + "WENB[109]" + "WENB[108]" + "WENB[107]" + 
   "WENB[106]" + "WENB[105]" + "WENB[104]" + "WENB[103]" + "WENB[102]" + "WENB[101]" + 
   "WENB[100]" + "WENB[99]" + "WENB[98]" + "WENB[97]" + "WENB[96]" + "WENB[95]" + 
   "WENB[94]" + "WENB[93]" + "WENB[92]" + "WENB[91]" + "WENB[90]" + "WENB[89]" + 
   "WENB[88]" + "WENB[87]" + "WENB[86]" + "WENB[85]" + "WENB[84]" + "WENB[83]" + 
   "WENB[82]" + "WENB[81]" + "WENB[80]" + "WENB[79]" + "WENB[78]" + "WENB[77]" + 
   "WENB[76]" + "WENB[75]" + "WENB[74]" + "WENB[73]" + "WENB[72]" + "WENB[71]" + 
   "WENB[70]" + "WENB[69]" + "WENB[68]" + "WENB[67]" + "WENB[66]" + "WENB[65]" + 
   "WENB[64]" + "WENB[63]" + "WENB[62]" + "WENB[61]" + "WENB[60]" + "WENB[59]" + 
   "WENB[58]" + "WENB[57]" + "WENB[56]" + "WENB[55]" + "WENB[54]" + "WENB[53]" + 
   "WENB[52]" + "WENB[51]" + "WENB[50]" + "WENB[49]" + "WENB[48]" + "WENB[47]" + 
   "WENB[46]" + "WENB[45]" + "WENB[44]" + "WENB[43]" + "WENB[42]" + "WENB[41]" + 
   "WENB[40]" + "WENB[39]" + "WENB[38]" + "WENB[37]" + "WENB[36]" + "WENB[35]" + 
   "WENB[34]" + "WENB[33]" + "WENB[32]" + "WENB[31]" + "WENB[30]" + "WENB[29]" + 
   "WENB[28]" + "WENB[27]" + "WENB[26]" + "WENB[25]" + "WENB[24]" + "WENB[23]" + 
   "WENB[22]" + "WENB[21]" + "WENB[20]" + "WENB[19]" + "WENB[18]" + "WENB[17]" + 
   "WENB[16]" + "WENB[15]" + "WENB[14]" + "WENB[13]" + "WENB[12]" + "WENB[11]" + 
   "WENB[10]" + "WENB[9]" + "WENB[8]" + "WENB[7]" + "WENB[6]" + "WENB[5]" + "WENB[4]" + 
   "WENB[3]" + "WENB[2]" + "WENB[1]" + "WENB[0]" + "AB[7]" + "AB[6]" + "AB[5]" + 
   "AB[4]" + "AB[3]" + "AB[2]" + "AB[1]" + "AB[0]" + "DB[127]" + "DB[126]" + "DB[125]" + 
   "DB[124]" + "DB[123]" + "DB[122]" + "DB[121]" + "DB[120]" + "DB[119]" + "DB[118]" + 
   "DB[117]" + "DB[116]" + "DB[115]" + "DB[114]" + "DB[113]" + "DB[112]" + "DB[111]" + 
   "DB[110]" + "DB[109]" + "DB[108]" + "DB[107]" + "DB[106]" + "DB[105]" + "DB[104]" + 
   "DB[103]" + "DB[102]" + "DB[101]" + "DB[100]" + "DB[99]" + "DB[98]" + "DB[97]" + 
   "DB[96]" + "DB[95]" + "DB[94]" + "DB[93]" + "DB[92]" + "DB[91]" + "DB[90]" + "DB[89]" + 
   "DB[88]" + "DB[87]" + "DB[86]" + "DB[85]" + "DB[84]" + "DB[83]" + "DB[82]" + "DB[81]" + 
   "DB[80]" + "DB[79]" + "DB[78]" + "DB[77]" + "DB[76]" + "DB[75]" + "DB[74]" + "DB[73]" + 
   "DB[72]" + "DB[71]" + "DB[70]" + "DB[69]" + "DB[68]" + "DB[67]" + "DB[66]" + "DB[65]" + 
   "DB[64]" + "DB[63]" + "DB[62]" + "DB[61]" + "DB[60]" + "DB[59]" + "DB[58]" + "DB[57]" + 
   "DB[56]" + "DB[55]" + "DB[54]" + "DB[53]" + "DB[52]" + "DB[51]" + "DB[50]" + "DB[49]" + 
   "DB[48]" + "DB[47]" + "DB[46]" + "DB[45]" + "DB[44]" + "DB[43]" + "DB[42]" + "DB[41]" + 
   "DB[40]" + "DB[39]" + "DB[38]" + "DB[37]" + "DB[36]" + "DB[35]" + "DB[34]" + "DB[33]" + 
   "DB[32]" + "DB[31]" + "DB[30]" + "DB[29]" + "DB[28]" + "DB[27]" + "DB[26]" + "DB[25]" + 
   "DB[24]" + "DB[23]" + "DB[22]" + "DB[21]" + "DB[20]" + "DB[19]" + "DB[18]" + "DB[17]" + 
   "DB[16]" + "DB[15]" + "DB[14]" + "DB[13]" + "DB[12]" + "DB[11]" + "DB[10]" + "DB[9]" + 
   "DB[8]" + "DB[7]" + "DB[6]" + "DB[5]" + "DB[4]" + "DB[3]" + "DB[2]" + "DB[1]" + 
   "DB[0]" + "EMAA[2]" + "EMAA[1]" + "EMAA[0]" + "EMASA" + "EMAB[2]" + "EMAB[1]" + 
   "EMAB[0]" + "TENA" + "TCENA" + "TAA[7]" + "TAA[6]" + "TAA[5]" + "TAA[4]" + "TAA[3]" + 
   "TAA[2]" + "TAA[1]" + "TAA[0]" + "TENB" + "TCENB" + "TWENB[127]" + "TWENB[126]" + 
   "TWENB[125]" + "TWENB[124]" + "TWENB[123]" + "TWENB[122]" + "TWENB[121]" + "TWENB[120]" + 
   "TWENB[119]" + "TWENB[118]" + "TWENB[117]" + "TWENB[116]" + "TWENB[115]" + "TWENB[114]" + 
   "TWENB[113]" + "TWENB[112]" + "TWENB[111]" + "TWENB[110]" + "TWENB[109]" + "TWENB[108]" + 
   "TWENB[107]" + "TWENB[106]" + "TWENB[105]" + "TWENB[104]" + "TWENB[103]" + "TWENB[102]" + 
   "TWENB[101]" + "TWENB[100]" + "TWENB[99]" + "TWENB[98]" + "TWENB[97]" + "TWENB[96]" + 
   "TWENB[95]" + "TWENB[94]" + "TWENB[93]" + "TWENB[92]" + "TWENB[91]" + "TWENB[90]" + 
   "TWENB[89]" + "TWENB[88]" + "TWENB[87]" + "TWENB[86]" + "TWENB[85]" + "TWENB[84]" + 
   "TWENB[83]" + "TWENB[82]" + "TWENB[81]" + "TWENB[80]" + "TWENB[79]" + "TWENB[78]" + 
   "TWENB[77]" + "TWENB[76]" + "TWENB[75]" + "TWENB[74]" + "TWENB[73]" + "TWENB[72]" + 
   "TWENB[71]" + "TWENB[70]" + "TWENB[69]" + "TWENB[68]" + "TWENB[67]" + "TWENB[66]" + 
   "TWENB[65]" + "TWENB[64]" + "TWENB[63]" + "TWENB[62]" + "TWENB[61]" + "TWENB[60]" + 
   "TWENB[59]" + "TWENB[58]" + "TWENB[57]" + "TWENB[56]" + "TWENB[55]" + "TWENB[54]" + 
   "TWENB[53]" + "TWENB[52]" + "TWENB[51]" + "TWENB[50]" + "TWENB[49]" + "TWENB[48]" + 
   "TWENB[47]" + "TWENB[46]" + "TWENB[45]" + "TWENB[44]" + "TWENB[43]" + "TWENB[42]" + 
   "TWENB[41]" + "TWENB[40]" + "TWENB[39]" + "TWENB[38]" + "TWENB[37]" + "TWENB[36]" + 
   "TWENB[35]" + "TWENB[34]" + "TWENB[33]" + "TWENB[32]" + "TWENB[31]" + "TWENB[30]" + 
   "TWENB[29]" + "TWENB[28]" + "TWENB[27]" + "TWENB[26]" + "TWENB[25]" + "TWENB[24]" + 
   "TWENB[23]" + "TWENB[22]" + "TWENB[21]" + "TWENB[20]" + "TWENB[19]" + "TWENB[18]" + 
   "TWENB[17]" + "TWENB[16]" + "TWENB[15]" + "TWENB[14]" + "TWENB[13]" + "TWENB[12]" + 
   "TWENB[11]" + "TWENB[10]" + "TWENB[9]" + "TWENB[8]" + "TWENB[7]" + "TWENB[6]" + 
   "TWENB[5]" + "TWENB[4]" + "TWENB[3]" + "TWENB[2]" + "TWENB[1]" + "TWENB[0]" + 
   "TAB[7]" + "TAB[6]" + "TAB[5]" + "TAB[4]" + "TAB[3]" + "TAB[2]" + "TAB[1]" + "TAB[0]" + 
   "TDB[127]" + "TDB[126]" + "TDB[125]" + "TDB[124]" + "TDB[123]" + "TDB[122]" + 
   "TDB[121]" + "TDB[120]" + "TDB[119]" + "TDB[118]" + "TDB[117]" + "TDB[116]" + 
   "TDB[115]" + "TDB[114]" + "TDB[113]" + "TDB[112]" + "TDB[111]" + "TDB[110]" + 
   "TDB[109]" + "TDB[108]" + "TDB[107]" + "TDB[106]" + "TDB[105]" + "TDB[104]" + 
   "TDB[103]" + "TDB[102]" + "TDB[101]" + "TDB[100]" + "TDB[99]" + "TDB[98]" + "TDB[97]" + 
   "TDB[96]" + "TDB[95]" + "TDB[94]" + "TDB[93]" + "TDB[92]" + "TDB[91]" + "TDB[90]" + 
   "TDB[89]" + "TDB[88]" + "TDB[87]" + "TDB[86]" + "TDB[85]" + "TDB[84]" + "TDB[83]" + 
   "TDB[82]" + "TDB[81]" + "TDB[80]" + "TDB[79]" + "TDB[78]" + "TDB[77]" + "TDB[76]" + 
   "TDB[75]" + "TDB[74]" + "TDB[73]" + "TDB[72]" + "TDB[71]" + "TDB[70]" + "TDB[69]" + 
   "TDB[68]" + "TDB[67]" + "TDB[66]" + "TDB[65]" + "TDB[64]" + "TDB[63]" + "TDB[62]" + 
   "TDB[61]" + "TDB[60]" + "TDB[59]" + "TDB[58]" + "TDB[57]" + "TDB[56]" + "TDB[55]" + 
   "TDB[54]" + "TDB[53]" + "TDB[52]" + "TDB[51]" + "TDB[50]" + "TDB[49]" + "TDB[48]" + 
   "TDB[47]" + "TDB[46]" + "TDB[45]" + "TDB[44]" + "TDB[43]" + "TDB[42]" + "TDB[41]" + 
   "TDB[40]" + "TDB[39]" + "TDB[38]" + "TDB[37]" + "TDB[36]" + "TDB[35]" + "TDB[34]" + 
   "TDB[33]" + "TDB[32]" + "TDB[31]" + "TDB[30]" + "TDB[29]" + "TDB[28]" + "TDB[27]" + 
   "TDB[26]" + "TDB[25]" + "TDB[24]" + "TDB[23]" + "TDB[22]" + "TDB[21]" + "TDB[20]" + 
   "TDB[19]" + "TDB[18]" + "TDB[17]" + "TDB[16]" + "TDB[15]" + "TDB[14]" + "TDB[13]" + 
   "TDB[12]" + "TDB[11]" + "TDB[10]" + "TDB[9]" + "TDB[8]" + "TDB[7]" + "TDB[6]" + 
   "TDB[5]" + "TDB[4]" + "TDB[3]" + "TDB[2]" + "TDB[1]" + "TDB[0]" + "RET1N" + "SIA[1]" + 
   "SIA[0]" + "SEA" + "DFTRAMBYP" + "SIB[1]" + "SIB[0]" + "SEB" + "COLLDISN"';
   "_po" = '"CENYA" + "AYA[7]" + "AYA[6]" + "AYA[5]" + "AYA[4]" + "AYA[3]" + "AYA[2]" + 
   "AYA[1]" + "AYA[0]" + "CENYB" + "WENYB[127]" + "WENYB[126]" + "WENYB[125]" + "WENYB[124]" + 
   "WENYB[123]" + "WENYB[122]" + "WENYB[121]" + "WENYB[120]" + "WENYB[119]" + "WENYB[118]" + 
   "WENYB[117]" + "WENYB[116]" + "WENYB[115]" + "WENYB[114]" + "WENYB[113]" + "WENYB[112]" + 
   "WENYB[111]" + "WENYB[110]" + "WENYB[109]" + "WENYB[108]" + "WENYB[107]" + "WENYB[106]" + 
   "WENYB[105]" + "WENYB[104]" + "WENYB[103]" + "WENYB[102]" + "WENYB[101]" + "WENYB[100]" + 
   "WENYB[99]" + "WENYB[98]" + "WENYB[97]" + "WENYB[96]" + "WENYB[95]" + "WENYB[94]" + 
   "WENYB[93]" + "WENYB[92]" + "WENYB[91]" + "WENYB[90]" + "WENYB[89]" + "WENYB[88]" + 
   "WENYB[87]" + "WENYB[86]" + "WENYB[85]" + "WENYB[84]" + "WENYB[83]" + "WENYB[82]" + 
   "WENYB[81]" + "WENYB[80]" + "WENYB[79]" + "WENYB[78]" + "WENYB[77]" + "WENYB[76]" + 
   "WENYB[75]" + "WENYB[74]" + "WENYB[73]" + "WENYB[72]" + "WENYB[71]" + "WENYB[70]" + 
   "WENYB[69]" + "WENYB[68]" + "WENYB[67]" + "WENYB[66]" + "WENYB[65]" + "WENYB[64]" + 
   "WENYB[63]" + "WENYB[62]" + "WENYB[61]" + "WENYB[60]" + "WENYB[59]" + "WENYB[58]" + 
   "WENYB[57]" + "WENYB[56]" + "WENYB[55]" + "WENYB[54]" + "WENYB[53]" + "WENYB[52]" + 
   "WENYB[51]" + "WENYB[50]" + "WENYB[49]" + "WENYB[48]" + "WENYB[47]" + "WENYB[46]" + 
   "WENYB[45]" + "WENYB[44]" + "WENYB[43]" + "WENYB[42]" + "WENYB[41]" + "WENYB[40]" + 
   "WENYB[39]" + "WENYB[38]" + "WENYB[37]" + "WENYB[36]" + "WENYB[35]" + "WENYB[34]" + 
   "WENYB[33]" + "WENYB[32]" + "WENYB[31]" + "WENYB[30]" + "WENYB[29]" + "WENYB[28]" + 
   "WENYB[27]" + "WENYB[26]" + "WENYB[25]" + "WENYB[24]" + "WENYB[23]" + "WENYB[22]" + 
   "WENYB[21]" + "WENYB[20]" + "WENYB[19]" + "WENYB[18]" + "WENYB[17]" + "WENYB[16]" + 
   "WENYB[15]" + "WENYB[14]" + "WENYB[13]" + "WENYB[12]" + "WENYB[11]" + "WENYB[10]" + 
   "WENYB[9]" + "WENYB[8]" + "WENYB[7]" + "WENYB[6]" + "WENYB[5]" + "WENYB[4]" + 
   "WENYB[3]" + "WENYB[2]" + "WENYB[1]" + "WENYB[0]" + "AYB[7]" + "AYB[6]" + "AYB[5]" + 
   "AYB[4]" + "AYB[3]" + "AYB[2]" + "AYB[1]" + "AYB[0]" + "QA[127]" + "QA[126]" + 
   "QA[125]" + "QA[124]" + "QA[123]" + "QA[122]" + "QA[121]" + "QA[120]" + "QA[119]" + 
   "QA[118]" + "QA[117]" + "QA[116]" + "QA[115]" + "QA[114]" + "QA[113]" + "QA[112]" + 
   "QA[111]" + "QA[110]" + "QA[109]" + "QA[108]" + "QA[107]" + "QA[106]" + "QA[105]" + 
   "QA[104]" + "QA[103]" + "QA[102]" + "QA[101]" + "QA[100]" + "QA[99]" + "QA[98]" + 
   "QA[97]" + "QA[96]" + "QA[95]" + "QA[94]" + "QA[93]" + "QA[92]" + "QA[91]" + "QA[90]" + 
   "QA[89]" + "QA[88]" + "QA[87]" + "QA[86]" + "QA[85]" + "QA[84]" + "QA[83]" + "QA[82]" + 
   "QA[81]" + "QA[80]" + "QA[79]" + "QA[78]" + "QA[77]" + "QA[76]" + "QA[75]" + "QA[74]" + 
   "QA[73]" + "QA[72]" + "QA[71]" + "QA[70]" + "QA[69]" + "QA[68]" + "QA[67]" + "QA[66]" + 
   "QA[65]" + "QA[64]" + "QA[63]" + "QA[62]" + "QA[61]" + "QA[60]" + "QA[59]" + "QA[58]" + 
   "QA[57]" + "QA[56]" + "QA[55]" + "QA[54]" + "QA[53]" + "QA[52]" + "QA[51]" + "QA[50]" + 
   "QA[49]" + "QA[48]" + "QA[47]" + "QA[46]" + "QA[45]" + "QA[44]" + "QA[43]" + "QA[42]" + 
   "QA[41]" + "QA[40]" + "QA[39]" + "QA[38]" + "QA[37]" + "QA[36]" + "QA[35]" + "QA[34]" + 
   "QA[33]" + "QA[32]" + "QA[31]" + "QA[30]" + "QA[29]" + "QA[28]" + "QA[27]" + "QA[26]" + 
   "QA[25]" + "QA[24]" + "QA[23]" + "QA[22]" + "QA[21]" + "QA[20]" + "QA[19]" + "QA[18]" + 
   "QA[17]" + "QA[16]" + "QA[15]" + "QA[14]" + "QA[13]" + "QA[12]" + "QA[11]" + "QA[10]" + 
   "QA[9]" + "QA[8]" + "QA[7]" + "QA[6]" + "QA[5]" + "QA[4]" + "QA[3]" + "QA[2]" + 
   "QA[1]" + "QA[0]" + "SOA[1]" + "SOA[0]" + "SOB[1]" + "SOB[0]"';
   "_si" = '"SIA[0]" + "SIA[1]" + "SIB[0]" + "SIB[1]"' {ScanIn; }
   "_so" = '"SOA[0]" + "SOA[1]" + "SOB[0]" + "SOB[1]"' {ScanOut; }
}
ScanStructures {
   ScanChain "chain_rf2_256x128_wm1_1" {
      ScanLength  64;
      ScanCells   "uDQA63" "uDQA62" "uDQA61" "uDQA60" "uDQA59" "uDQA58" "uDQA57" "uDQA56" "uDQA55" "uDQA54" "uDQA53" "uDQA52" "uDQA51" "uDQA50" "uDQA49" "uDQA48" "uDQA47" "uDQA46" "uDQA45" "uDQA44" "uDQA43" "uDQA42" "uDQA41" "uDQA40" "uDQA39" "uDQA38" "uDQA37" "uDQA36" "uDQA35" "uDQA34" "uDQA33" "uDQA32" "uDQA31" "uDQA30" "uDQA29" "uDQA28" "uDQA27" "uDQA26" "uDQA25" "uDQA24" "uDQA23" "uDQA22" "uDQA21" "uDQA20" "uDQA19" "uDQA18" "uDQA17" "uDQA16" "uDQA15" "uDQA14" "uDQA13" "uDQA12" "uDQA11" "uDQA10" "uDQA9" "uDQA8" "uDQA7" "uDQA6" "uDQA5" "uDQA4" "uDQA3" "uDQA2" "uDQA1" "uDQA0" ;
      ScanIn  "SIA[0]";
      ScanOut  "SOA[0]";
      ScanEnable  "SEA";
      ScanMasterClock  "CLKA";
   }
   ScanChain "chain_rf2_256x128_wm1_2" {
      ScanLength  64;
      ScanCells  "uDQA64" "uDQA65" "uDQA66" "uDQA67" "uDQA68" "uDQA69" "uDQA70" "uDQA71" "uDQA72" "uDQA73" "uDQA74" "uDQA75" "uDQA76" "uDQA77" "uDQA78" "uDQA79" "uDQA80" "uDQA81" "uDQA82" "uDQA83" "uDQA84" "uDQA85" "uDQA86" "uDQA87" "uDQA88" "uDQA89" "uDQA90" "uDQA91" "uDQA92" "uDQA93" "uDQA94" "uDQA95" "uDQA96" "uDQA97" "uDQA98" "uDQA99" "uDQA100" "uDQA101" "uDQA102" "uDQA103" "uDQA104" "uDQA105" "uDQA106" "uDQA107" "uDQA108" "uDQA109" "uDQA110" "uDQA111" "uDQA112" "uDQA113" "uDQA114" "uDQA115" "uDQA116" "uDQA117" "uDQA118" "uDQA119" "uDQA120" "uDQA121" "uDQA122" "uDQA123" "uDQA124" "uDQA125" "uDQA126" "uDQA127"  ;
      ScanIn  "SIA[1]";
      ScanOut  "SOA[1]";
      ScanEnable  "SEA";
      ScanMasterClock  "CLKA";
   }
   ScanChain "chain_rf2_256x128_wm1_3" {
      ScanLength  64;
      ScanCells   "uDQB63" "uDQB62" "uDQB61" "uDQB60" "uDQB59" "uDQB58" "uDQB57" "uDQB56" "uDQB55" "uDQB54" "uDQB53" "uDQB52" "uDQB51" "uDQB50" "uDQB49" "uDQB48" "uDQB47" "uDQB46" "uDQB45" "uDQB44" "uDQB43" "uDQB42" "uDQB41" "uDQB40" "uDQB39" "uDQB38" "uDQB37" "uDQB36" "uDQB35" "uDQB34" "uDQB33" "uDQB32" "uDQB31" "uDQB30" "uDQB29" "uDQB28" "uDQB27" "uDQB26" "uDQB25" "uDQB24" "uDQB23" "uDQB22" "uDQB21" "uDQB20" "uDQB19" "uDQB18" "uDQB17" "uDQB16" "uDQB15" "uDQB14" "uDQB13" "uDQB12" "uDQB11" "uDQB10" "uDQB9" "uDQB8" "uDQB7" "uDQB6" "uDQB5" "uDQB4" "uDQB3" "uDQB2" "uDQB1" "uDQB0" ;
      ScanIn  "SIB[0]";
      ScanOut  "SOB[0]";
      ScanEnable  "SEB";
      ScanMasterClock  "CLKB";
   }
   ScanChain "chain_rf2_256x128_wm1_4" {
      ScanLength  64;
      ScanCells  "uDQB64" "uDQB65" "uDQB66" "uDQB67" "uDQB68" "uDQB69" "uDQB70" "uDQB71" "uDQB72" "uDQB73" "uDQB74" "uDQB75" "uDQB76" "uDQB77" "uDQB78" "uDQB79" "uDQB80" "uDQB81" "uDQB82" "uDQB83" "uDQB84" "uDQB85" "uDQB86" "uDQB87" "uDQB88" "uDQB89" "uDQB90" "uDQB91" "uDQB92" "uDQB93" "uDQB94" "uDQB95" "uDQB96" "uDQB97" "uDQB98" "uDQB99" "uDQB100" "uDQB101" "uDQB102" "uDQB103" "uDQB104" "uDQB105" "uDQB106" "uDQB107" "uDQB108" "uDQB109" "uDQB110" "uDQB111" "uDQB112" "uDQB113" "uDQB114" "uDQB115" "uDQB116" "uDQB117" "uDQB118" "uDQB119" "uDQB120" "uDQB121" "uDQB122" "uDQB123" "uDQB124" "uDQB125" "uDQB126" "uDQB127"  ;
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
Environment "rf2_256x128_wm1" {
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
