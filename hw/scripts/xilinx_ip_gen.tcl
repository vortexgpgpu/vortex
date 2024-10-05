# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if { $::argc < 1 || $::argc > 2 } {
    puts "ERROR: Program \"$::argv0\" requires 1 or 2 arguments!\n"
    puts "Usage: $::argv0 <ip_dir> [<device_part>]\n"
    exit
}

set ip_dir [lindex $::argv 0]

# create_ip requires that a project is open in memory.
if { $::argc == 2 } {
    set device_part [lindex $::argv 1]
    create_project -in_memory -part $device_part
} else {
    # Create project without specifying a device part
    create_project -in_memory
}

# IP folder does not exist. Create IP folder
file mkdir ${ip_dir}

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name xil_fdiv -dir ${ip_dir}
set_property -dict [list CONFIG.Component_Name {xil_fdiv} CONFIG.Operation_Type {Divide} CONFIG.Flow_Control {NonBlocking} CONFIG.Has_ACLKEN {true} CONFIG.C_Has_UNDERFLOW {true} CONFIG.C_Has_OVERFLOW {true} CONFIG.C_Has_INVALID_OP {true} CONFIG.C_Has_DIVIDE_BY_ZERO {true} CONFIG.A_Precision_Type {Single} CONFIG.C_A_Exponent_Width {8} CONFIG.C_A_Fraction_Width {24} CONFIG.Result_Precision_Type {Single} CONFIG.C_Result_Exponent_Width {8} CONFIG.C_Result_Fraction_Width {24} CONFIG.C_Mult_Usage {No_Usage} CONFIG.Has_RESULT_TREADY {false} CONFIG.C_Latency {28} CONFIG.C_Rate {1}] [get_ips xil_fdiv]

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name xil_fsqrt -dir ${ip_dir}
set_property -dict [list CONFIG.Component_Name {xil_fsqrt} CONFIG.Operation_Type {Square_root} CONFIG.Flow_Control {NonBlocking} CONFIG.Has_ACLKEN {true} CONFIG.C_Has_INVALID_OP {true} CONFIG.A_Precision_Type {Single} CONFIG.C_A_Exponent_Width {8} CONFIG.C_A_Fraction_Width {24} CONFIG.Result_Precision_Type {Single} CONFIG.C_Result_Exponent_Width {8} CONFIG.C_Result_Fraction_Width {24} CONFIG.C_Mult_Usage {No_Usage} CONFIG.Has_RESULT_TREADY {false} CONFIG.C_Latency {28} CONFIG.C_Rate {1}] [get_ips xil_fsqrt]

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name xil_fma -dir ${ip_dir}
set_property -dict [list CONFIG.Component_Name {xil_fma} CONFIG.Operation_Type {FMA} CONFIG.Add_Sub_Value {Add} CONFIG.Flow_Control {NonBlocking} CONFIG.Has_ACLKEN {true} CONFIG.C_Has_UNDERFLOW {true} CONFIG.C_Has_OVERFLOW {true} CONFIG.C_Has_INVALID_OP {true} CONFIG.Has_A_TUSER {false} CONFIG.A_Precision_Type {Single} CONFIG.C_A_Exponent_Width {8} CONFIG.C_A_Fraction_Width {24} CONFIG.Result_Precision_Type {Single} CONFIG.C_Result_Exponent_Width {8} CONFIG.C_Result_Fraction_Width {24} CONFIG.C_Mult_Usage {Medium_Usage} CONFIG.Has_RESULT_TREADY {false} CONFIG.C_Latency {16} CONFIG.C_Rate {1} CONFIG.A_TUSER_Width {1}] [get_ips xil_fma]

generate_target all [get_ips]

close_project -delete
