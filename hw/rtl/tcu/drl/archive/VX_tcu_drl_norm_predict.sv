// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

// Wrt bf16 mul,
// Norm shift is required if, a * b ≥ 32,768 (16th bit of result is set)
// therefore, a ≥ 32,768 / b
// storing all possible values of threshold (32,768 / b) in LUT

module VX_tcu_drl_norm_predict (
    input wire [7:0] b_input,
    output logic [7:0] threshold
);

    always_comb begin
        case (b_input)
            8'd128: threshold = 8'd255;
            8'd129: threshold = 8'd255;
            8'd130: threshold = 8'd253;
            8'd131: threshold = 8'd251;
            8'd132: threshold = 8'd249;
            8'd133: threshold = 8'd247;
            8'd134: threshold = 8'd245;
            8'd135: threshold = 8'd243;
            8'd136: threshold = 8'd241;
            8'd137: threshold = 8'd240;
            8'd138: threshold = 8'd238;
            8'd139: threshold = 8'd236;
            8'd140: threshold = 8'd235;
            8'd141: threshold = 8'd233;
            8'd142: threshold = 8'd231;
            8'd143: threshold = 8'd230;
            8'd144: threshold = 8'd228;
            8'd145: threshold = 8'd226;
            8'd146: threshold = 8'd225;
            8'd147: threshold = 8'd223;
            8'd148: threshold = 8'd222;
            8'd149: threshold = 8'd220;
            8'd150: threshold = 8'd219;
            8'd151: threshold = 8'd218;
            8'd152: threshold = 8'd216;
            8'd153: threshold = 8'd215;
            8'd154: threshold = 8'd213;
            8'd155: threshold = 8'd212;
            8'd156: threshold = 8'd211;
            8'd157: threshold = 8'd209;
            8'd158: threshold = 8'd208;
            8'd159: threshold = 8'd207;
            8'd160: threshold = 8'd205;
            8'd161: threshold = 8'd204;
            8'd162: threshold = 8'd203;
            8'd163: threshold = 8'd202;
            8'd164: threshold = 8'd200;
            8'd165: threshold = 8'd199;
            8'd166: threshold = 8'd198;
            8'd167: threshold = 8'd197;
            8'd168: threshold = 8'd196;
            8'd169: threshold = 8'd194;
            8'd170: threshold = 8'd193;
            8'd171: threshold = 8'd192;
            8'd172: threshold = 8'd191;
            8'd173: threshold = 8'd190;
            8'd174: threshold = 8'd189;
            8'd175: threshold = 8'd188;
            8'd176: threshold = 8'd187;
            8'd177: threshold = 8'd186;
            8'd178: threshold = 8'd185;
            8'd179: threshold = 8'd184;
            8'd180: threshold = 8'd183;
            8'd181: threshold = 8'd182;
            8'd182: threshold = 8'd181;
            8'd183: threshold = 8'd180;
            8'd184: threshold = 8'd179;
            8'd185: threshold = 8'd178;
            8'd186: threshold = 8'd177;
            8'd187: threshold = 8'd176;
            8'd188: threshold = 8'd175;
            8'd189: threshold = 8'd174;
            8'd190: threshold = 8'd173;
            8'd191: threshold = 8'd172;
            8'd192: threshold = 8'd171;
            8'd193: threshold = 8'd170;
            8'd194: threshold = 8'd169;
            8'd195: threshold = 8'd169;
            8'd196: threshold = 8'd168;
            8'd197: threshold = 8'd167;
            8'd198: threshold = 8'd166;
            8'd199: threshold = 8'd165;
            8'd200: threshold = 8'd164;
            8'd201: threshold = 8'd164;
            8'd202: threshold = 8'd163;
            8'd203: threshold = 8'd162;
            8'd204: threshold = 8'd161;
            8'd205: threshold = 8'd160;
            8'd206: threshold = 8'd160;
            8'd207: threshold = 8'd159;
            8'd208: threshold = 8'd158;
            8'd209: threshold = 8'd157;
            8'd210: threshold = 8'd157;
            8'd211: threshold = 8'd156;
            8'd212: threshold = 8'd155;
            8'd213: threshold = 8'd154;
            8'd214: threshold = 8'd154;
            8'd215: threshold = 8'd153;
            8'd216: threshold = 8'd152;
            8'd217: threshold = 8'd152;
            8'd218: threshold = 8'd151;
            8'd219: threshold = 8'd150;
            8'd220: threshold = 8'd149;
            8'd221: threshold = 8'd149;
            8'd222: threshold = 8'd148;
            8'd223: threshold = 8'd147;
            8'd224: threshold = 8'd147;
            8'd225: threshold = 8'd146;
            8'd226: threshold = 8'd145;
            8'd227: threshold = 8'd145;
            8'd228: threshold = 8'd144;
            8'd229: threshold = 8'd144;
            8'd230: threshold = 8'd143;
            8'd231: threshold = 8'd142;
            8'd232: threshold = 8'd142;
            8'd233: threshold = 8'd141;
            8'd234: threshold = 8'd141;
            8'd235: threshold = 8'd140;
            8'd236: threshold = 8'd139;
            8'd237: threshold = 8'd139;
            8'd238: threshold = 8'd138;
            8'd239: threshold = 8'd138;
            8'd240: threshold = 8'd137;
            8'd241: threshold = 8'd136;
            8'd242: threshold = 8'd136;
            8'd243: threshold = 8'd135;
            8'd244: threshold = 8'd135;
            8'd245: threshold = 8'd134;
            8'd246: threshold = 8'd134;
            8'd247: threshold = 8'd133;
            8'd248: threshold = 8'd133;
            8'd249: threshold = 8'd132;
            8'd250: threshold = 8'd132;
            8'd251: threshold = 8'd131;
            8'd252: threshold = 8'd131;
            8'd253: threshold = 8'd130;
            8'd254: threshold = 8'd130;
            8'd255: threshold = 8'd129;
            default: threshold = 8'd255;
        endcase
    end

endmodule
