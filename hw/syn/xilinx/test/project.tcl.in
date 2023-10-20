if { $::argc != 3 } {
    puts "ERROR: Program \"$::argv0\" requires 3 arguments!\n"
    puts "Usage: $::argv0 <vcs_file> <files_dir> <tool_dir>\n"
    exit
}

set vcs_file [lindex $::argv 0]
set files_dir [lindex $::argv 1]
set tool_dir [lindex $::argv 2]

#puts $vcs_file
#puts $files_dir
#puts $tool_dir

set origin_dir [file normalize "."]

# Use origin directory path location variable, if specified in the tcl shell
if { [info exists ::origin_dir_loc] } {
  set origin_dir $::origin_dir_loc
}

# Set the project name
set project_name "project_1"

# Use project name variable, if specified in the tcl shell
if { [info exists ::user_project_name] } {
  set project_name $::user_project_name
}

source "${tool_dir}/parse_vcs_list.tcl"
set vlist [parse_vcs_list "${vcs_file}"]

set vsources_list  [lindex $vlist 0]
set vincludes_list [lindex $vlist 1]
set vdefines_list  [lindex $vlist 2]

#puts ${vsources_list}
#puts ${vincludes_list}
#puts ${vdefines_list}

# Create project
create_project ${project_name} ./${project_name} -force -part xcu280-fsvh2892-2L-e

# Set the directory path for the new project
set proj_dir [get_property directory [current_project]]

# Set project properties
set obj [current_project]
set_property -name "board_part" -value "xilinx.com:au280:part0:1.1" -objects $obj
set_property -name "compxlib.activehdl_compiled_library_dir" -value "$proj_dir/${project_name}.cache/compile_simlib/activehdl" -objects $obj
set_property -name "compxlib.funcsim" -value "1" -objects $obj
set_property -name "compxlib.ies_compiled_library_dir" -value "$proj_dir/${project_name}.cache/compile_simlib/ies" -objects $obj
set_property -name "compxlib.modelsim_compiled_library_dir" -value "$proj_dir/${project_name}.cache/compile_simlib/modelsim" -objects $obj
set_property -name "compxlib.overwrite_libs" -value "0" -objects $obj
set_property -name "compxlib.questa_compiled_library_dir" -value "$proj_dir/${project_name}.cache/compile_simlib/questa" -objects $obj
set_property -name "compxlib.riviera_compiled_library_dir" -value "$proj_dir/${project_name}.cache/compile_simlib/riviera" -objects $obj
set_property -name "compxlib.timesim" -value "1" -objects $obj
set_property -name "compxlib.vcs_compiled_library_dir" -value "$proj_dir/${project_name}.cache/compile_simlib/vcs" -objects $obj
set_property -name "compxlib.xsim_compiled_library_dir" -value "" -objects $obj
set_property -name "corecontainer.enable" -value "0" -objects $obj
set_property -name "default_lib" -value "xil_defaultlib" -objects $obj
set_property -name "enable_optional_runs_sta" -value "0" -objects $obj
set_property -name "enable_vhdl_2008" -value "1" -objects $obj
set_property -name "generate_ip_upgrade_log" -value "1" -objects $obj
set_property -name "ip_cache_permissions" -value "read write" -objects $obj
set_property -name "ip_interface_inference_priority" -value "" -objects $obj
set_property -name "ip_output_repo" -value "$proj_dir/${project_name}.cache/ip" -objects $obj
set_property -name "legacy_ip_repo_paths" -value "" -objects $obj
set_property -name "mem.enable_memory_map_generation" -value "1" -objects $obj
set_property -name "platform.board_id" -value "au280" -objects $obj
set_property -name "platform.default_output_type" -value "undefined" -objects $obj
set_property -name "platform.design_intent.datacenter" -value "undefined" -objects $obj
set_property -name "platform.design_intent.embedded" -value "undefined" -objects $obj
set_property -name "platform.design_intent.external_host" -value "undefined" -objects $obj
set_property -name "platform.design_intent.server_managed" -value "undefined" -objects $obj
set_property -name "platform.rom.debug_type" -value "0" -objects $obj
set_property -name "platform.rom.prom_type" -value "0" -objects $obj
set_property -name "platform.slrconstraintmode" -value "0" -objects $obj
set_property -name "preferred_sim_model" -value "rtl" -objects $obj
set_property -name "project_type" -value "Default" -objects $obj
set_property -name "pr_flow" -value "0" -objects $obj
set_property -name "sim.central_dir" -value "$proj_dir/${project_name}.ip_user_files" -objects $obj
set_property -name "sim.ip.auto_export_scripts" -value "1" -objects $obj
set_property -name "sim.use_ip_compiled_libs" -value "1" -objects $obj
set_property -name "simulator.activehdl_gcc_install_dir" -value "" -objects $obj
set_property -name "simulator.activehdl_install_dir" -value "" -objects $obj
set_property -name "simulator.ies_gcc_install_dir" -value "" -objects $obj
set_property -name "simulator.ies_install_dir" -value "" -objects $obj
set_property -name "simulator.modelsim_gcc_install_dir" -value "" -objects $obj
set_property -name "simulator.modelsim_install_dir" -value "" -objects $obj
set_property -name "simulator.questa_gcc_install_dir" -value "" -objects $obj
set_property -name "simulator.riviera_gcc_install_dir" -value "" -objects $obj
set_property -name "simulator.riviera_install_dir" -value "" -objects $obj
set_property -name "simulator.vcs_gcc_install_dir" -value "" -objects $obj
set_property -name "simulator.vcs_install_dir" -value "" -objects $obj
set_property -name "simulator.xcelium_gcc_install_dir" -value "" -objects $obj
set_property -name "simulator.xcelium_install_dir" -value "" -objects $obj
set_property -name "simulator_language" -value "Verilog" -objects $obj
set_property -name "source_mgmt_mode" -value "All" -objects $obj
set_property -name "target_language" -value "Verilog" -objects $obj
set_property -name "target_simulator" -value "XSim" -objects $obj
set_property -name "tool_flow" -value "Vivado" -objects $obj
set_property -name "webtalk.activehdl_export_sim" -value "27" -objects $obj
set_property -name "webtalk.ies_export_sim" -value "27" -objects $obj
set_property -name "webtalk.modelsim_export_sim" -value "27" -objects $obj
set_property -name "webtalk.questa_export_sim" -value "27" -objects $obj
set_property -name "webtalk.riviera_export_sim" -value "27" -objects $obj
set_property -name "webtalk.vcs_export_sim" -value "27" -objects $obj
set_property -name "webtalk.xcelium_export_sim" -value "5" -objects $obj
set_property -name "webtalk.xsim_export_sim" -value "27" -objects $obj
set_property -name "webtalk.xsim_launch_sim" -value "91" -objects $obj
set_property -name "xpm_libraries" -value "XPM_CDC XPM_MEMORY" -objects $obj
set_property -name "xsim.array_display_limit" -value "1024" -objects $obj
set_property -name "xsim.radix" -value "hex" -objects $obj
set_property -name "xsim.time_unit" -value "ns" -objects $obj
set_property -name "xsim.trace_limit" -value "65536" -objects $obj

# Create 'sources_1' fileset (if not found)
if {[string equal [get_filesets -quiet sources_1] ""]} {
  create_fileset -srcset sources_1
}

# add source files
set obj [get_filesets sources_1]
add_files -norecurse -verbose -fileset $obj ${vsources_list}

# process defines
set obj [get_filesets sources_1]
foreach def $vdefines_list {
  set_property -name "verilog_define" -value $def -objects $obj
}

# Set 'sources_1' fileset properties
set obj [get_filesets sources_1]
set_property -name "design_mode" -value "RTL" -objects $obj
set_property -name "edif_extra_search_paths" -value "" -objects $obj
set_property -name "elab_link_dcps" -value "1" -objects $obj
set_property -name "elab_load_timing_constraints" -value "1" -objects $obj
set_property -name "generic" -value "" -objects $obj
set_property -name "include_dirs" -value "" -objects $obj
set_property -name "lib_map_file" -value "" -objects $obj
set_property -name "loop_count" -value "1000" -objects $obj
set_property -name "name" -value "sources_1" -objects $obj
set_property -name "top" -value "design_1_wrapper" -objects $obj
set_property -name "top_auto_set" -value "0" -objects $obj
set_property -name "verilog_define" -value "" -objects $obj
set_property -name "verilog_uppercase" -value "1" -objects $obj
set_property -name "verilog_version" -value "verilog_2001" -objects $obj
set_property -name "vhdl_version" -value "vhdl_2k" -objects $obj

# Create 'constrs_1' fileset (if not found)
if {[string equal [get_filesets -quiet constrs_1] ""]} {
  create_fileset -constrset constrs_1
}

# Set 'constrs_1' fileset object
set obj [get_filesets constrs_1]

# Empty (no sources present)

# Set 'constrs_1' fileset properties
set obj [get_filesets constrs_1]
set_property -name "constrs_type" -value "XDC" -objects $obj
set_property -name "name" -value "constrs_1" -objects $obj
set_property -name "target_constrs_file" -value "" -objects $obj

# Create 'sim_1' fileset (if not found)
if {[string equal [get_filesets -quiet sim_1] ""]} {
  create_fileset -simset sim_1
}

# Set 'sim_1' fileset object
set obj [get_filesets sim_1]
# Import local files from the original project
set files [list \
 [file normalize "$files_dir/testbench.v" ]\
]
set imported_files [import_files -fileset sim_1 $files]

# Set 'sim_1' fileset file properties for remote files
# None

# Set 'sim_1' fileset file properties for local files
set file "testbench.v"
set file_obj [get_files -of_objects [get_filesets sim_1] [list "*$file"]]
set_property -name "file_type" -value "Verilog" -objects $file_obj
set_property -name "is_enabled" -value "1" -objects $file_obj
set_property -name "is_global_include" -value "0" -objects $file_obj
set_property -name "library" -value "xil_defaultlib" -objects $file_obj
set_property -name "path_mode" -value "RelativeFirst" -objects $file_obj
set_property -name "used_in" -value "synthesis implementation simulation" -objects $file_obj
set_property -name "used_in_implementation" -value "1" -objects $file_obj
set_property -name "used_in_simulation" -value "1" -objects $file_obj
set_property -name "used_in_synthesis" -value "1" -objects $file_obj

# Set 'sim_1' fileset properties
set obj [get_filesets sim_1]
set_property -name "32bit" -value "0" -objects $obj
set_property -name "force_compile_glbl" -value "0" -objects $obj
set_property -name "generate_scripts_only" -value "0" -objects $obj
set_property -name "generic" -value "" -objects $obj
set_property -name "hbs.configure_design_for_hier_access" -value "1" -objects $obj
set_property -name "include_dirs" -value "" -objects $obj
set_property -name "incremental" -value "1" -objects $obj
set_property -name "name" -value "sim_1" -objects $obj
set_property -name "nl.cell" -value "" -objects $obj
set_property -name "nl.incl_unisim_models" -value "0" -objects $obj
set_property -name "nl.mode" -value "funcsim" -objects $obj
set_property -name "nl.process_corner" -value "slow" -objects $obj
set_property -name "nl.rename_top" -value "" -objects $obj
set_property -name "nl.sdf_anno" -value "1" -objects $obj
set_property -name "nl.write_all_overrides" -value "0" -objects $obj
set_property -name "source_set" -value "sources_1" -objects $obj
set_property -name "systemc_include_dirs" -value "" -objects $obj
set_property -name "top" -value "testbench" -objects $obj
set_property -name "top_auto_set" -value "0" -objects $obj
set_property -name "top_lib" -value "xil_defaultlib" -objects $obj
set_property -name "transport_int_delay" -value "0" -objects $obj
set_property -name "transport_path_delay" -value "0" -objects $obj
set_property -name "unifast" -value "0" -objects $obj
set_property -name "verilog_define" -value "" -objects $obj
set_property -name "verilog_uppercase" -value "0" -objects $obj
set_property -name "xelab.dll" -value "0" -objects $obj
set_property -name "xsim.compile.tcl.pre" -value "" -objects $obj
set_property -name "xsim.compile.xsc.more_options" -value "" -objects $obj
set_property -name "xsim.compile.xvhdl.more_options" -value "" -objects $obj
set_property -name "xsim.compile.xvhdl.nosort" -value "1" -objects $obj
set_property -name "xsim.compile.xvhdl.relax" -value "1" -objects $obj
set_property -name "xsim.compile.xvlog.more_options" -value "" -objects $obj
set_property -name "xsim.compile.xvlog.nosort" -value "1" -objects $obj
set_property -name "xsim.compile.xvlog.relax" -value "1" -objects $obj
set_property -name "xsim.elaborate.debug_level" -value "typical" -objects $obj
set_property -name "xsim.elaborate.load_glbl" -value "1" -objects $obj
set_property -name "xsim.elaborate.mt_level" -value "auto" -objects $obj
set_property -name "xsim.elaborate.rangecheck" -value "0" -objects $obj
set_property -name "xsim.elaborate.relax" -value "1" -objects $obj
set_property -name "xsim.elaborate.sdf_delay" -value "sdfmax" -objects $obj
set_property -name "xsim.elaborate.snapshot" -value "" -objects $obj
set_property -name "xsim.elaborate.xelab.more_options" -value "" -objects $obj
set_property -name "xsim.elaborate.xsc.more_options" -value "" -objects $obj
set_property -name "xsim.simulate.add_positional" -value "0" -objects $obj
set_property -name "xsim.simulate.custom_tcl" -value "" -objects $obj
set_property -name "xsim.simulate.log_all_signals" -value "0" -objects $obj
set_property -name "xsim.simulate.no_quit" -value "0" -objects $obj
set_property -name "xsim.simulate.runtime" -value "4000ns" -objects $obj
set_property -name "xsim.simulate.saif" -value "" -objects $obj
set_property -name "xsim.simulate.saif_all_signals" -value "0" -objects $obj
set_property -name "xsim.simulate.saif_scope" -value "" -objects $obj
set_property -name "xsim.simulate.tcl.post" -value "" -objects $obj
set_property -name "xsim.simulate.wdb" -value "" -objects $obj
set_property -name "xsim.simulate.xsim.more_options" -value "" -objects $obj

# Set 'utils_1' fileset object
set obj [get_filesets utils_1]
# Empty (no sources present)

# Set 'utils_1' fileset properties
set obj [get_filesets utils_1]
set_property -name "name" -value "utils_1" -objects $obj

# Proc to create BD design_1
proc cr_bd_design_1 { parentCell } {
# The design that will be created by this Tcl proc contains the following 
# module references:
# Vortex_top

# CHANGE DESIGN NAME HERE
set design_name design_1

common::send_gid_msg -ssname BD::TCL -id 2010 -severity "INFO" "Currently there is no design <$design_name> in project, so creating one..."

create_bd_design $design_name

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
     set list_check_ips "\ 
  xilinx.com:ip:axi_bram_ctrl:4.1\
  xilinx.com:ip:blk_mem_gen:8.4\
  "

   set list_ips_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2011 -severity "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2012 -severity "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

  }

  ##################################################################
  # CHECK Modules
  ##################################################################
  set bCheckModules 1
  if { $bCheckModules == 1 } {
     set list_check_mods "\ 
  Vortex_top\
  "

   set list_mods_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2020 -severity "INFO" "Checking if the following modules exist in the project's sources: $list_check_mods ."

   foreach mod_vlnv $list_check_mods {
      if { [can_resolve_reference $mod_vlnv] == 0 } {
         lappend list_mods_missing $mod_vlnv
      }
   }

   if { $list_mods_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2021 -severity "ERROR" "The following module(s) are not found in the project: $list_mods_missing" }
      common::send_gid_msg -ssname BD::TCL -id 2022 -severity "INFO" "Please add source files for the missing module(s) above."
      set bCheckIPsPassed 0
   }
}

if { $bCheckIPsPassed != 1 } {
  common::send_gid_msg -ssname BD::TCL -id 2023 -severity "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

variable script_folder

if { $parentCell eq "" } {
    set parentCell [get_bd_cells /]
}

# Get object for parentCell
set parentObj [get_bd_cells $parentCell]
if { $parentObj == "" } {
    catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
    return
}

# Make sure parentObj is hier blk
set parentType [get_property TYPE $parentObj]
if { $parentType ne "hier" } {
    catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
    return
}

# Save current instance; Restore later
set oldCurInst [current_bd_instance .]

# Set parent object as current
current_bd_instance $parentObj


# Create interface ports

# Create ports
set clk_100MHz [ create_bd_port -dir I -type clk -freq_hz 100000000 clk_100MHz ]
set resetn [ create_bd_port -dir I -type rst resetn ]
set_property -dict [ list \
  CONFIG.POLARITY {ACTIVE_LOW} \
] $resetn
set vx_busy [ create_bd_port -dir O vx_busy ]
set vx_reset [ create_bd_port -dir I -type rst vx_reset ]
set_property -dict [ list \
  CONFIG.POLARITY {ACTIVE_HIGH} \
] $vx_reset
 
set dcr_wr_valid [ create_bd_port -dir I dcr_wr_valid ]
set dcr_wr_addr [ create_bd_port -dir I -from 11 -to 0 dcr_wr_addr ]
set dcr_wr_data [ create_bd_port -dir I -from 31 -to 0 dcr_wr_data ]

# Create instance: Vortex_top_0, and set properties
set block_name Vortex_top
set block_cell_name Vortex_top_0
if { [catch {set Vortex_top_0 [create_bd_cell -type module -reference $block_name $block_cell_name] } errmsg] } {
    catch {common::send_gid_msg -ssname BD::TCL -id 2095 -severity "ERROR" "Unable to add referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
    return 1
  } elseif { $Vortex_top_0 eq "" } {
    catch {common::send_gid_msg -ssname BD::TCL -id 2096 -severity "ERROR" "Unable to referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
    return 1
  }
  
# Create instance: axi_bram_ctrl_0, and set properties
set axi_bram_ctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0 ]
set_property -dict [ list \
  CONFIG.DATA_WIDTH {512} \
  CONFIG.ECC_TYPE {0} \
] $axi_bram_ctrl_0

# Create instance: axi_bram_ctrl_0_bram, and set properties
set axi_bram_ctrl_0_bram [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 axi_bram_ctrl_0_bram ]

set_property -dict [ list \
  CONFIG.Assume_Synchronous_Clk {true} \
  CONFIG.Byte_Size {8} \
  CONFIG.Load_Init_File {true} \
  CONFIG.Coe_File {%COE_FILE%} \
  CONFIG.EN_SAFETY_CKT {true} \
  CONFIG.Enable_32bit_Address {true} \
  CONFIG.Fill_Remaining_Memory_Locations {false} \
  CONFIG.Memory_Type {Simple_Dual_Port_RAM} \
  CONFIG.Operating_Mode_A {NO_CHANGE} \
  CONFIG.Operating_Mode_B {READ_FIRST} \
  CONFIG.Port_B_Write_Rate {0} \
  CONFIG.Read_Width_A {512} \
  CONFIG.Read_Width_B {512} \
  CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
  CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
  CONFIG.Remaining_Memory_Locations {0} \
  CONFIG.Use_Byte_Write_Enable {true} \
  CONFIG.Use_RSTA_Pin {false} \
  CONFIG.Use_RSTB_Pin {true} \
  CONFIG.Write_Width_A {512} \
  CONFIG.Write_Depth_A {16384} \
  CONFIG.use_bram_block {Stand_Alone} \
] $axi_bram_ctrl_0_bram

# Create interface connections
connect_bd_intf_net -intf_net Vortex_top_0_m_axi_mem [get_bd_intf_pins Vortex_top_0/m_axi_mem] [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]
connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTA [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins axi_bram_ctrl_0_bram/BRAM_PORTA]
connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTB [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTB] [get_bd_intf_pins axi_bram_ctrl_0_bram/BRAM_PORTB]

# Create port connections
connect_bd_net -net Vortex_top_0_busy [get_bd_ports vx_busy] [get_bd_pins Vortex_top_0/busy]
connect_bd_net -net clk_wiz_clk_out1 [get_bd_ports clk_100MHz] [get_bd_pins Vortex_top_0/clk] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk]
connect_bd_net -net resetn_1 [get_bd_ports resetn] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn]
connect_bd_net -net vx_reset_1 [get_bd_ports vx_reset] [get_bd_pins Vortex_top_0/reset]
connect_bd_net -net dcr_wr_valid_1 [get_bd_ports dcr_wr_valid] [get_bd_pins Vortex_top_0/dcr_wr_valid]
connect_bd_net -net dcr_wr_addr_1 [get_bd_ports dcr_wr_addr] [get_bd_pins Vortex_top_0/dcr_wr_addr]
connect_bd_net -net dcr_wr_data_1 [get_bd_ports dcr_wr_data] [get_bd_pins Vortex_top_0/dcr_wr_data]

# Create address segments
assign_bd_address -offset 0x00000000 -range 0x00100000 -target_address_space [get_bd_addr_spaces Vortex_top_0/m_axi_mem] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force

# Perform GUI Layout
regenerate_bd_layout -layout_string {
  "ActiveEmotionalView":"Default View",
  "Default View_ScaleFactor":"1.0",
  "Default View_TopLeft":"-195,-165",
  "ExpandedHierarchyInLayout":"",
  "guistr":"# # String gsaved with Nlview 7.0r4  2019-12-20 bk=1.5203 VDI=41 GEI=36 GUI=JA:10.0 TLS
#  -string -flagsOSRD
preplace port clk_100MHz -pg 1 -lvl 0 -x 0 -y 40 -defaultsOSRD
preplace port resetn -pg 1 -lvl 0 -x 0 -y 20 -defaultsOSRD
preplace port vx_busy -pg 1 -lvl 4 -x 950 -y 220 -defaultsOSRD
preplace port vx_reset -pg 1 -lvl 0 -x 0 -y 110 -defaultsOSRD
preplace port dcr_wr_valid -pg 1 -lvl 0 -x 0 -y 130 -defaultsOSRD
preplace portBus dcr_wr_addr -pg 1 -lvl 0 -x 0 -y 150 -defaultsOSRD
preplace portBus dcr_wr_data -pg 1 -lvl 0 -x 0 -y 170 -defaultsOSRD
preplace inst Vortex_top_0 -pg 1 -lvl 1 -x 190 -y 130 -defaultsOSRD
preplace inst axi_bram_ctrl_0 -pg 1 -lvl 2 -x 520 -y 140 -defaultsOSRD
preplace inst axi_bram_ctrl_0_bram -pg 1 -lvl 3 -x 800 -y 140 -defaultsOSRD
preplace netloc Vortex_top_0_busy 1 1 3 360J 220 NJ 220 NJ
preplace netloc clk_wiz_clk_out1 1 0 2 20 30 370
preplace netloc resetn_1 1 0 2 NJ 20 380J
preplace netloc vx_reset_1 1 0 1 NJ 110
preplace netloc dcr_wr_valid_1 1 0 1 NJ 130
preplace netloc dcr_wr_addr_1 1 0 1 NJ 150
preplace netloc dcr_wr_data_1 1 0 1 NJ 170
preplace netloc axi_bram_ctrl_0_BRAM_PORTB 1 2 1 N 150
preplace netloc axi_bram_ctrl_0_BRAM_PORTA 1 2 1 N 130
preplace netloc Vortex_top_0_m_axi_mem 1 1 1 N 120
levelinfo -pg 1 0 190 520 800 950
pagesize -pg 1 -db -bbox -sgen -180 0 1060 240
"
}

  # Restore current instance
  current_bd_instance $oldCurInst

  validate_bd_design
  save_bd_design
  close_bd_design $design_name 
}
# End of cr_bd_design_1()
cr_bd_design_1 ""
set_property EXCLUDE_DEBUG_LOGIC "0" [get_files design_1.bd ] 
set_property GENERATE_SYNTH_CHECKPOINT "1" [get_files design_1.bd ] 
set_property IS_ENABLED "1" [get_files design_1.bd ] 
set_property IS_GLOBAL_INCLUDE "0" [get_files design_1.bd ] 
#set_property IS_LOCKED "0" [get_files design_1.bd ] 
set_property LIBRARY "xil_defaultlib" [get_files design_1.bd ] 
set_property PATH_MODE "RelativeFirst" [get_files design_1.bd ] 
set_property PFM_NAME "" [get_files design_1.bd ] 
set_property REGISTERED_WITH_MANAGER "1" [get_files design_1.bd ] 
set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [get_files design_1.bd ] 
set_property USED_IN "synthesis implementation simulation" [get_files design_1.bd ] 
set_property USED_IN_IMPLEMENTATION "1" [get_files design_1.bd ] 
set_property USED_IN_SIMULATION "1" [get_files design_1.bd ] 
set_property USED_IN_SYNTHESIS "1" [get_files design_1.bd ] 

#call make_wrapper to create wrapper files
set wrapper_path [make_wrapper -fileset sources_1 -files [ get_files -norecurse design_1.bd] -top]
add_files -norecurse -fileset sources_1 $wrapper_path

# Create 'synth_1' run (if not found)
if {[string equal [get_runs -quiet synth_1] ""]} {
    create_run -name synth_1 -part xcu280-fsvh2892-2L-e -flow {Vivado Synthesis 2020} -strategy "Vivado Synthesis Defaults" -report_strategy {No Reports} -constrset constrs_1
} else {
  set_property strategy "Vivado Synthesis Defaults" [get_runs synth_1]
  set_property flow "Vivado Synthesis 2020" [get_runs synth_1]
}
set obj [get_runs synth_1]
set_property set_report_strategy_name 1 $obj
set_property report_strategy {Vivado Synthesis Default Reports} $obj
set_property set_report_strategy_name 0 $obj
# Create 'synth_1_synth_report_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs synth_1] synth_1_synth_report_utilization_0] "" ] } {
  create_report_config -report_name synth_1_synth_report_utilization_0 -report_type report_utilization:1.0 -steps synth_design -runs synth_1
}
set obj [get_report_configs -of_objects [get_runs synth_1] synth_1_synth_report_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Utilization - Synth Design" -objects $obj
set_property -name "options.pblocks" -value "" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.slr" -value "0" -objects $obj
set_property -name "options.packthru" -value "0" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.hierarchical_percentages" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
set obj [get_runs synth_1]
set_property -name "constrset" -value "constrs_1" -objects $obj
set_property -name "description" -value "Vivado Synthesis Defaults" -objects $obj
set_property -name "flow" -value "Vivado Synthesis 2020" -objects $obj
set_property -name "name" -value "synth_1" -objects $obj
set_property -name "needs_refresh" -value "0" -objects $obj
set_property -name "srcset" -value "sources_1" -objects $obj
set_property -name "incremental_checkpoint" -value "" -objects $obj
set_property -name "auto_incremental_checkpoint" -value "0" -objects $obj
set_property -name "rqs_files" -value "" -objects $obj
set_property -name "incremental_checkpoint.more_options" -value "" -objects $obj
set_property -name "include_in_archive" -value "1" -objects $obj
set_property -name "gen_full_bitstream" -value "1" -objects $obj
set_property -name "write_incremental_synth_checkpoint" -value "0" -objects $obj
set_property -name "auto_incremental_checkpoint.directory" -value "$proj_dir/project_1.srcs/utils_1/imports/synth_1" -objects $obj
set_property -name "strategy" -value "Vivado Synthesis Defaults" -objects $obj
set_property -name "steps.synth_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.synth_design.tcl.post" -value "" -objects $obj
set_property -name "steps.synth_design.args.flatten_hierarchy" -value "rebuilt" -objects $obj
set_property -name "steps.synth_design.args.gated_clock_conversion" -value "off" -objects $obj
set_property -name "steps.synth_design.args.bufg" -value "12" -objects $obj
set_property -name "steps.synth_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.synth_design.args.retiming" -value "0" -objects $obj
set_property -name "steps.synth_design.args.fsm_extraction" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.keep_equivalent_registers" -value "0" -objects $obj
set_property -name "steps.synth_design.args.resource_sharing" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.control_set_opt_threshold" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.no_lc" -value "0" -objects $obj
set_property -name "steps.synth_design.args.no_srlextract" -value "0" -objects $obj
set_property -name "steps.synth_design.args.shreg_min_size" -value "3" -objects $obj
set_property -name "steps.synth_design.args.max_bram" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_uram" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_dsp" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_bram_cascade_height" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_uram_cascade_height" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.cascade_dsp" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.assert" -value "0" -objects $obj
set_property -name "steps.synth_design.args.more options" -value "" -objects $obj

# Create 'synth_1_copy_1' run (if not found)
if {[string equal [get_runs -quiet synth_1_copy_1] ""]} {
    create_run -name synth_1_copy_1 -part xcu280-fsvh2892-2L-e -flow {Vivado Synthesis 2020} -strategy "Vivado Synthesis Defaults" -report_strategy {No Reports} -constrset constrs_1
} else {
  set_property strategy "Vivado Synthesis Defaults" [get_runs synth_1_copy_1]
  set_property flow "Vivado Synthesis 2020" [get_runs synth_1_copy_1]
}
set obj [get_runs synth_1_copy_1]
set_property set_report_strategy_name 1 $obj
set_property report_strategy {Vivado Synthesis Default Reports} $obj
set_property set_report_strategy_name 0 $obj
# Create 'synth_1_copy_1_synth_report_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs synth_1_copy_1] synth_1_copy_1_synth_report_utilization_0] "" ] } {
  create_report_config -report_name synth_1_copy_1_synth_report_utilization_0 -report_type report_utilization:1.0 -steps synth_design -runs synth_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs synth_1_copy_1] synth_1_copy_1_synth_report_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Utilization - Synth Design" -objects $obj
set_property -name "options.pblocks" -value "" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.slr" -value "0" -objects $obj
set_property -name "options.packthru" -value "0" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.hierarchical_percentages" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
set obj [get_runs synth_1_copy_1]
set_property -name "constrset" -value "constrs_1" -objects $obj
set_property -name "description" -value "Vivado Synthesis Defaults" -objects $obj
set_property -name "flow" -value "Vivado Synthesis 2020" -objects $obj
set_property -name "name" -value "synth_1_copy_1" -objects $obj
set_property -name "needs_refresh" -value "0" -objects $obj
set_property -name "srcset" -value "sources_1" -objects $obj
set_property -name "incremental_checkpoint" -value "" -objects $obj
set_property -name "auto_incremental_checkpoint" -value "0" -objects $obj
set_property -name "rqs_files" -value "" -objects $obj
set_property -name "incremental_checkpoint.more_options" -value "" -objects $obj
set_property -name "include_in_archive" -value "1" -objects $obj
set_property -name "gen_full_bitstream" -value "1" -objects $obj
set_property -name "write_incremental_synth_checkpoint" -value "0" -objects $obj
set_property -name "auto_incremental_checkpoint.directory" -value "$proj_dir/project_1.srcs/utils_1/imports/synth_1" -objects $obj
set_property -name "strategy" -value "Vivado Synthesis Defaults" -objects $obj
set_property -name "steps.synth_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.synth_design.tcl.post" -value "" -objects $obj
set_property -name "steps.synth_design.args.flatten_hierarchy" -value "rebuilt" -objects $obj
set_property -name "steps.synth_design.args.gated_clock_conversion" -value "off" -objects $obj
set_property -name "steps.synth_design.args.bufg" -value "12" -objects $obj
set_property -name "steps.synth_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.synth_design.args.retiming" -value "0" -objects $obj
set_property -name "steps.synth_design.args.fsm_extraction" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.keep_equivalent_registers" -value "0" -objects $obj
set_property -name "steps.synth_design.args.resource_sharing" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.control_set_opt_threshold" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.no_lc" -value "0" -objects $obj
set_property -name "steps.synth_design.args.no_srlextract" -value "0" -objects $obj
set_property -name "steps.synth_design.args.shreg_min_size" -value "3" -objects $obj
set_property -name "steps.synth_design.args.max_bram" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_uram" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_dsp" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_bram_cascade_height" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.max_uram_cascade_height" -value "-1" -objects $obj
set_property -name "steps.synth_design.args.cascade_dsp" -value "auto" -objects $obj
set_property -name "steps.synth_design.args.assert" -value "0" -objects $obj
set_property -name "steps.synth_design.args.more options" -value "" -objects $obj

# set the current synth run
current_run -synthesis [get_runs synth_1]

# preserve signal names
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none [get_runs synth_1]

# Create 'impl_1' run (if not found)
if {[string equal [get_runs -quiet impl_1] ""]} {
    create_run -name impl_1 -part xcu280-fsvh2892-2L-e -flow {Vivado Implementation 2020} -strategy "Vivado Implementation Defaults" -report_strategy {No Reports} -constrset constrs_1 -parent_run synth_1
} else {
  set_property strategy "Vivado Implementation Defaults" [get_runs impl_1]
  set_property flow "Vivado Implementation 2020" [get_runs impl_1]
}
set obj [get_runs impl_1]
set_property set_report_strategy_name 1 $obj
set_property report_strategy {Vivado Implementation Default Reports} $obj
set_property set_report_strategy_name 0 $obj
# Create 'impl_1_init_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_init_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_init_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps init_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_init_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Design Initialization" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_opt_report_drc_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_opt_report_drc_0] "" ] } {
  create_report_config -report_name impl_1_opt_report_drc_0 -report_type report_drc:1.0 -steps opt_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_opt_report_drc_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "DRC - Opt Design" -objects $obj
set_property -name "options.upgrade_cw" -value "0" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.ruledecks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps opt_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_power_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_power_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_power_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps power_opt_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_power_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Power Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_place_report_io_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_io_0] "" ] } {
  create_report_config -report_name impl_1_place_report_io_0 -report_type report_io:1.0 -steps place_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_io_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "IO - Place Design" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_place_report_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_utilization_0] "" ] } {
  create_report_config -report_name impl_1_place_report_utilization_0 -report_type report_utilization:1.0 -steps place_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Utilization - Place Design" -objects $obj
set_property -name "options.pblocks" -value "" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.slr" -value "0" -objects $obj
set_property -name "options.packthru" -value "0" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.hierarchical_percentages" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_place_report_control_sets_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_control_sets_0] "" ] } {
  create_report_config -report_name impl_1_place_report_control_sets_0 -report_type report_control_sets:1.0 -steps place_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_control_sets_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Control Sets - Place Design" -objects $obj
set_property -name "options.verbose" -value "1" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_place_report_incremental_reuse_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_incremental_reuse_0] "" ] } {
  create_report_config -report_name impl_1_place_report_incremental_reuse_0 -report_type report_incremental_reuse:1.0 -steps place_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_incremental_reuse_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Place Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_place_report_incremental_reuse_1' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_incremental_reuse_1] "" ] } {
  create_report_config -report_name impl_1_place_report_incremental_reuse_1 -report_type report_incremental_reuse:1.0 -steps place_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_incremental_reuse_1]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Place Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_place_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_place_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps place_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_place_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Place Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_post_place_power_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_post_place_power_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_post_place_power_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps post_place_power_opt_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_post_place_power_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Place Power Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_phys_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_phys_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_phys_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps phys_opt_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_phys_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Place Phys Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_drc_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_drc_0] "" ] } {
  create_report_config -report_name impl_1_route_report_drc_0 -report_type report_drc:1.0 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_drc_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "DRC - Route Design" -objects $obj
set_property -name "options.upgrade_cw" -value "0" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.ruledecks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_methodology_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_methodology_0] "" ] } {
  create_report_config -report_name impl_1_route_report_methodology_0 -report_type report_methodology:1.0 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_methodology_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Methodology - Route Design" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_power_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_power_0] "" ] } {
  create_report_config -report_name impl_1_route_report_power_0 -report_type report_power:1.0 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_power_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Power - Route Design" -objects $obj
set_property -name "options.advisory" -value "0" -objects $obj
set_property -name "options.xpe" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_route_status_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_route_status_0] "" ] } {
  create_report_config -report_name impl_1_route_report_route_status_0 -report_type report_route_status:1.0 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_route_status_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Route Status - Route Design" -objects $obj
set_property -name "options.of_objects" -value "" -objects $obj
set_property -name "options.route_type" -value "" -objects $obj
set_property -name "options.list_all_nets" -value "0" -objects $obj
set_property -name "options.show_all" -value "0" -objects $obj
set_property -name "options.has_routing" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_route_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Timing Summary - Route Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_incremental_reuse_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_incremental_reuse_0] "" ] } {
  create_report_config -report_name impl_1_route_report_incremental_reuse_0 -report_type report_incremental_reuse:1.0 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_incremental_reuse_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Route Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_clock_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_clock_utilization_0] "" ] } {
  create_report_config -report_name impl_1_route_report_clock_utilization_0 -report_type report_clock_utilization:1.0 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_clock_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Clock Utilization - Route Design" -objects $obj
set_property -name "options.write_xdc" -value "0" -objects $obj
set_property -name "options.clock_roots_only" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_route_report_bus_skew_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_bus_skew_0] "" ] } {
  create_report_config -report_name impl_1_route_report_bus_skew_0 -report_type report_bus_skew:1.1 -steps route_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_route_report_bus_skew_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Bus Skew - Route Design" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.slack_greater_than" -value "" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_post_route_phys_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_post_route_phys_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_post_route_phys_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps post_route_phys_opt_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_post_route_phys_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Route Phys Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_post_route_phys_opt_report_bus_skew_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1] impl_1_post_route_phys_opt_report_bus_skew_0] "" ] } {
  create_report_config -report_name impl_1_post_route_phys_opt_report_bus_skew_0 -report_type report_bus_skew:1.1 -steps post_route_phys_opt_design -runs impl_1
}
set obj [get_report_configs -of_objects [get_runs impl_1] impl_1_post_route_phys_opt_report_bus_skew_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Bus Skew - Post-Route Phys Opt Design" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.slack_greater_than" -value "" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
set obj [get_runs impl_1]
set_property -name "constrset" -value "constrs_1" -objects $obj
set_property -name "description" -value "Default settings for Implementation." -objects $obj
set_property -name "flow" -value "Vivado Implementation 2020" -objects $obj
set_property -name "name" -value "impl_1" -objects $obj
set_property -name "needs_refresh" -value "0" -objects $obj
set_property -name "pr_configuration" -value "" -objects $obj
set_property -name "srcset" -value "sources_1" -objects $obj
set_property -name "incremental_checkpoint" -value "" -objects $obj
set_property -name "auto_incremental_checkpoint" -value "0" -objects $obj
set_property -name "rqs_files" -value "" -objects $obj
set_property -name "incremental_checkpoint.more_options" -value "" -objects $obj
set_property -name "include_in_archive" -value "1" -objects $obj
set_property -name "gen_full_bitstream" -value "1" -objects $obj
set_property -name "auto_incremental_checkpoint.directory" -value "$proj_dir/project_1.srcs/utils_1/imports/impl_1" -objects $obj
set_property -name "strategy" -value "Vivado Implementation Defaults" -objects $obj
set_property -name "steps.init_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.init_design.tcl.post" -value "" -objects $obj
set_property -name "steps.opt_design.is_enabled" -value "1" -objects $obj
set_property -name "steps.opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.opt_design.args.verbose" -value "0" -objects $obj
set_property -name "steps.opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.power_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.power_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.power_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.power_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.place_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.place_design.tcl.post" -value "" -objects $obj
set_property -name "steps.place_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.place_design.args.more options" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.post_place_power_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.phys_opt_design.is_enabled" -value "1" -objects $obj
set_property -name "steps.phys_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.phys_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.phys_opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.phys_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.route_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.route_design.tcl.post" -value "" -objects $obj
set_property -name "steps.route_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.route_design.args.more options" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.post_route_phys_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.post_route_phys_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.write_bitstream.tcl.pre" -value "" -objects $obj
set_property -name "steps.write_bitstream.tcl.post" -value "" -objects $obj
set_property -name "steps.write_bitstream.args.raw_bitfile" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.mask_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.no_binary_bitfile" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.bin_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.readback_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.logic_location_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.verbose" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.more options" -value "" -objects $obj

# Create 'impl_1_copy_1' run (if not found)
if {[string equal [get_runs -quiet impl_1_copy_1] ""]} {
    create_run -name impl_1_copy_1 -part xcu280-fsvh2892-2L-e -flow {Vivado Implementation 2020} -strategy "Vivado Implementation Defaults" -report_strategy {No Reports} -constrset constrs_1 -parent_run synth_1
} else {
  set_property strategy "Vivado Implementation Defaults" [get_runs impl_1_copy_1]
  set_property flow "Vivado Implementation 2020" [get_runs impl_1_copy_1]
}
set obj [get_runs impl_1_copy_1]
set_property set_report_strategy_name 1 $obj
set_property report_strategy {Vivado Implementation Default Reports} $obj
set_property set_report_strategy_name 0 $obj
# Create 'impl_1_copy_1_init_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_init_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_init_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps init_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_init_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Design Initialization" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_opt_report_drc_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_opt_report_drc_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_opt_report_drc_0 -report_type report_drc:1.0 -steps opt_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_opt_report_drc_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "DRC - Opt Design" -objects $obj
set_property -name "options.upgrade_cw" -value "0" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.ruledecks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps opt_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_power_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_power_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_power_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps power_opt_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_power_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Power Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_place_report_io_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_io_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_place_report_io_0 -report_type report_io:1.0 -steps place_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_io_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "IO - Place Design" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_place_report_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_utilization_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_place_report_utilization_0 -report_type report_utilization:1.0 -steps place_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Utilization - Place Design" -objects $obj
set_property -name "options.pblocks" -value "" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.slr" -value "0" -objects $obj
set_property -name "options.packthru" -value "0" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.hierarchical_percentages" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_place_report_control_sets_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_control_sets_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_place_report_control_sets_0 -report_type report_control_sets:1.0 -steps place_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_control_sets_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Control Sets - Place Design" -objects $obj
set_property -name "options.verbose" -value "1" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_place_report_incremental_reuse_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_incremental_reuse_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_place_report_incremental_reuse_0 -report_type report_incremental_reuse:1.0 -steps place_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_incremental_reuse_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Place Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_place_report_incremental_reuse_1' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_incremental_reuse_1] "" ] } {
  create_report_config -report_name impl_1_copy_1_place_report_incremental_reuse_1 -report_type report_incremental_reuse:1.0 -steps place_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_incremental_reuse_1]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Place Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_place_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_place_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps place_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_place_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Place Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_post_place_power_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_post_place_power_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_post_place_power_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps post_place_power_opt_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_post_place_power_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Place Power Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_phys_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_phys_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_phys_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps phys_opt_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_phys_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Place Phys Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_drc_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_drc_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_drc_0 -report_type report_drc:1.0 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_drc_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "DRC - Route Design" -objects $obj
set_property -name "options.upgrade_cw" -value "0" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.ruledecks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_methodology_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_methodology_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_methodology_0 -report_type report_methodology:1.0 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_methodology_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Methodology - Route Design" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_power_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_power_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_power_0 -report_type report_power:1.0 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_power_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Power - Route Design" -objects $obj
set_property -name "options.advisory" -value "0" -objects $obj
set_property -name "options.xpe" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_route_status_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_route_status_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_route_status_0 -report_type report_route_status:1.0 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_route_status_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Route Status - Route Design" -objects $obj
set_property -name "options.of_objects" -value "" -objects $obj
set_property -name "options.route_type" -value "" -objects $obj
set_property -name "options.list_all_nets" -value "0" -objects $obj
set_property -name "options.show_all" -value "0" -objects $obj
set_property -name "options.has_routing" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Timing Summary - Route Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_incremental_reuse_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_incremental_reuse_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_incremental_reuse_0 -report_type report_incremental_reuse:1.0 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_incremental_reuse_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Route Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_clock_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_clock_utilization_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_clock_utilization_0 -report_type report_clock_utilization:1.0 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_clock_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Clock Utilization - Route Design" -objects $obj
set_property -name "options.write_xdc" -value "0" -objects $obj
set_property -name "options.clock_roots_only" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_route_report_bus_skew_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_bus_skew_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_route_report_bus_skew_0 -report_type report_bus_skew:1.1 -steps route_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_route_report_bus_skew_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Bus Skew - Route Design" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.slack_greater_than" -value "" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_post_route_phys_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_post_route_phys_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_post_route_phys_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps post_route_phys_opt_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_post_route_phys_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Route Phys Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_1_post_route_phys_opt_report_bus_skew_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_post_route_phys_opt_report_bus_skew_0] "" ] } {
  create_report_config -report_name impl_1_copy_1_post_route_phys_opt_report_bus_skew_0 -report_type report_bus_skew:1.1 -steps post_route_phys_opt_design -runs impl_1_copy_1
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_1] impl_1_copy_1_post_route_phys_opt_report_bus_skew_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Bus Skew - Post-Route Phys Opt Design" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.slack_greater_than" -value "" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
set obj [get_runs impl_1_copy_1]
set_property -name "constrset" -value "constrs_1" -objects $obj
set_property -name "description" -value "Default settings for Implementation." -objects $obj
set_property -name "flow" -value "Vivado Implementation 2020" -objects $obj
set_property -name "name" -value "impl_1_copy_1" -objects $obj
set_property -name "needs_refresh" -value "0" -objects $obj
set_property -name "pr_configuration" -value "" -objects $obj
set_property -name "srcset" -value "sources_1" -objects $obj
set_property -name "incremental_checkpoint" -value "" -objects $obj
set_property -name "auto_incremental_checkpoint" -value "0" -objects $obj
set_property -name "rqs_files" -value "" -objects $obj
set_property -name "incremental_checkpoint.more_options" -value "" -objects $obj
set_property -name "include_in_archive" -value "1" -objects $obj
set_property -name "gen_full_bitstream" -value "1" -objects $obj
set_property -name "auto_incremental_checkpoint.directory" -value "$proj_dir/project_1.srcs/utils_1/imports/impl_1" -objects $obj
set_property -name "strategy" -value "Vivado Implementation Defaults" -objects $obj
set_property -name "steps.init_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.init_design.tcl.post" -value "" -objects $obj
set_property -name "steps.opt_design.is_enabled" -value "1" -objects $obj
set_property -name "steps.opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.opt_design.args.verbose" -value "0" -objects $obj
set_property -name "steps.opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.power_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.power_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.power_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.power_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.place_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.place_design.tcl.post" -value "" -objects $obj
set_property -name "steps.place_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.place_design.args.more options" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.post_place_power_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.phys_opt_design.is_enabled" -value "1" -objects $obj
set_property -name "steps.phys_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.phys_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.phys_opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.phys_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.route_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.route_design.tcl.post" -value "" -objects $obj
set_property -name "steps.route_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.route_design.args.more options" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.post_route_phys_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.post_route_phys_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.write_bitstream.tcl.pre" -value "" -objects $obj
set_property -name "steps.write_bitstream.tcl.post" -value "" -objects $obj
set_property -name "steps.write_bitstream.args.raw_bitfile" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.mask_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.no_binary_bitfile" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.bin_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.readback_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.logic_location_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.verbose" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.more options" -value "" -objects $obj

# Create 'impl_1_copy_2' run (if not found)
if {[string equal [get_runs -quiet impl_1_copy_2] ""]} {
    create_run -name impl_1_copy_2 -part xcu280-fsvh2892-2L-e -flow {Vivado Implementation 2020} -strategy "Vivado Implementation Defaults" -report_strategy {No Reports} -constrset constrs_1 -parent_run synth_1
} else {
  set_property strategy "Vivado Implementation Defaults" [get_runs impl_1_copy_2]
  set_property flow "Vivado Implementation 2020" [get_runs impl_1_copy_2]
}
set obj [get_runs impl_1_copy_2]
set_property set_report_strategy_name 1 $obj
set_property report_strategy {Vivado Implementation Default Reports} $obj
set_property set_report_strategy_name 0 $obj
# Create 'impl_1_copy_2_init_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_init_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_init_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps init_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_init_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Design Initialization" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_opt_report_drc_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_opt_report_drc_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_opt_report_drc_0 -report_type report_drc:1.0 -steps opt_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_opt_report_drc_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "DRC - Opt Design" -objects $obj
set_property -name "options.upgrade_cw" -value "0" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.ruledecks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps opt_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_power_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_power_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_power_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps power_opt_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_power_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Power Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_place_report_io_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_io_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_place_report_io_0 -report_type report_io:1.0 -steps place_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_io_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "IO - Place Design" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_place_report_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_utilization_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_place_report_utilization_0 -report_type report_utilization:1.0 -steps place_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Utilization - Place Design" -objects $obj
set_property -name "options.pblocks" -value "" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.slr" -value "0" -objects $obj
set_property -name "options.packthru" -value "0" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.hierarchical_percentages" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_place_report_control_sets_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_control_sets_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_place_report_control_sets_0 -report_type report_control_sets:1.0 -steps place_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_control_sets_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Control Sets - Place Design" -objects $obj
set_property -name "options.verbose" -value "1" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_place_report_incremental_reuse_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_incremental_reuse_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_place_report_incremental_reuse_0 -report_type report_incremental_reuse:1.0 -steps place_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_incremental_reuse_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Place Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_place_report_incremental_reuse_1' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_incremental_reuse_1] "" ] } {
  create_report_config -report_name impl_1_copy_2_place_report_incremental_reuse_1 -report_type report_incremental_reuse:1.0 -steps place_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_incremental_reuse_1]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Place Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_place_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_place_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps place_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_place_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Place Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_post_place_power_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_post_place_power_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_post_place_power_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps post_place_power_opt_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_post_place_power_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Place Power Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_phys_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_phys_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_phys_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps phys_opt_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_phys_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "0" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Place Phys Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_drc_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_drc_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_drc_0 -report_type report_drc:1.0 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_drc_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "DRC - Route Design" -objects $obj
set_property -name "options.upgrade_cw" -value "0" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.ruledecks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_methodology_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_methodology_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_methodology_0 -report_type report_methodology:1.0 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_methodology_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Methodology - Route Design" -objects $obj
set_property -name "options.checks" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_power_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_power_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_power_0 -report_type report_power:1.0 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_power_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Power - Route Design" -objects $obj
set_property -name "options.advisory" -value "0" -objects $obj
set_property -name "options.xpe" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_route_status_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_route_status_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_route_status_0 -report_type report_route_status:1.0 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_route_status_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Route Status - Route Design" -objects $obj
set_property -name "options.of_objects" -value "" -objects $obj
set_property -name "options.route_type" -value "" -objects $obj
set_property -name "options.list_all_nets" -value "0" -objects $obj
set_property -name "options.show_all" -value "0" -objects $obj
set_property -name "options.has_routing" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Timing Summary - Route Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "0" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_incremental_reuse_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_incremental_reuse_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_incremental_reuse_0 -report_type report_incremental_reuse:1.0 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_incremental_reuse_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Incremental Reuse - Route Design" -objects $obj
set_property -name "options.cells" -value "" -objects $obj
set_property -name "options.hierarchical" -value "0" -objects $obj
set_property -name "options.hierarchical_depth" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_clock_utilization_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_clock_utilization_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_clock_utilization_0 -report_type report_clock_utilization:1.0 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_clock_utilization_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Clock Utilization - Route Design" -objects $obj
set_property -name "options.write_xdc" -value "0" -objects $obj
set_property -name "options.clock_roots_only" -value "0" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_route_report_bus_skew_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_bus_skew_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_route_report_bus_skew_0 -report_type report_bus_skew:1.1 -steps route_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_route_report_bus_skew_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Bus Skew - Route Design" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.slack_greater_than" -value "" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_post_route_phys_opt_report_timing_summary_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_post_route_phys_opt_report_timing_summary_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_post_route_phys_opt_report_timing_summary_0 -report_type report_timing_summary:1.0 -steps post_route_phys_opt_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_post_route_phys_opt_report_timing_summary_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Timing Summary - Post-Route Phys Opt Design" -objects $obj
set_property -name "options.check_timing_verbose" -value "0" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "10" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.report_unconstrained" -value "0" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.cell" -value "" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
# Create 'impl_1_copy_2_post_route_phys_opt_report_bus_skew_0' report (if not found)
if { [ string equal [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_post_route_phys_opt_report_bus_skew_0] "" ] } {
  create_report_config -report_name impl_1_copy_2_post_route_phys_opt_report_bus_skew_0 -report_type report_bus_skew:1.1 -steps post_route_phys_opt_design -runs impl_1_copy_2
}
set obj [get_report_configs -of_objects [get_runs impl_1_copy_2] impl_1_copy_2_post_route_phys_opt_report_bus_skew_0]
if { $obj != "" } {
set_property -name "is_enabled" -value "1" -objects $obj
set_property -name "display_name" -value "Bus Skew - Post-Route Phys Opt Design" -objects $obj
set_property -name "options.delay_type" -value "" -objects $obj
set_property -name "options.setup" -value "0" -objects $obj
set_property -name "options.hold" -value "0" -objects $obj
set_property -name "options.max_paths" -value "" -objects $obj
set_property -name "options.nworst" -value "" -objects $obj
set_property -name "options.unique_pins" -value "0" -objects $obj
set_property -name "options.path_type" -value "" -objects $obj
set_property -name "options.slack_lesser_than" -value "" -objects $obj
set_property -name "options.slack_greater_than" -value "" -objects $obj
set_property -name "options.significant_digits" -value "" -objects $obj
set_property -name "options.warn_on_violation" -value "1" -objects $obj
set_property -name "options.more_options" -value "" -objects $obj

}
set obj [get_runs impl_1_copy_2]
set_property -name "constrset" -value "constrs_1" -objects $obj
set_property -name "description" -value "Default settings for Implementation." -objects $obj
set_property -name "flow" -value "Vivado Implementation 2020" -objects $obj
set_property -name "name" -value "impl_1_copy_2" -objects $obj
set_property -name "needs_refresh" -value "0" -objects $obj
set_property -name "pr_configuration" -value "" -objects $obj
set_property -name "srcset" -value "sources_1" -objects $obj
set_property -name "incremental_checkpoint" -value "" -objects $obj
set_property -name "auto_incremental_checkpoint" -value "0" -objects $obj
set_property -name "rqs_files" -value "" -objects $obj
set_property -name "incremental_checkpoint.more_options" -value "" -objects $obj
set_property -name "include_in_archive" -value "1" -objects $obj
set_property -name "gen_full_bitstream" -value "1" -objects $obj
set_property -name "auto_incremental_checkpoint.directory" -value "$proj_dir/project_1.srcs/utils_1/imports/impl_1" -objects $obj
set_property -name "strategy" -value "Vivado Implementation Defaults" -objects $obj
set_property -name "steps.init_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.init_design.tcl.post" -value "" -objects $obj
set_property -name "steps.opt_design.is_enabled" -value "1" -objects $obj
set_property -name "steps.opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.opt_design.args.verbose" -value "0" -objects $obj
set_property -name "steps.opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.power_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.power_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.power_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.power_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.place_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.place_design.tcl.post" -value "" -objects $obj
set_property -name "steps.place_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.place_design.args.more options" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.post_place_power_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.post_place_power_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.phys_opt_design.is_enabled" -value "1" -objects $obj
set_property -name "steps.phys_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.phys_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.phys_opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.phys_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.route_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.route_design.tcl.post" -value "" -objects $obj
set_property -name "steps.route_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.route_design.args.more options" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.is_enabled" -value "0" -objects $obj
set_property -name "steps.post_route_phys_opt_design.tcl.pre" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.tcl.post" -value "" -objects $obj
set_property -name "steps.post_route_phys_opt_design.args.directive" -value "Default" -objects $obj
set_property -name "steps.post_route_phys_opt_design.args.more options" -value "" -objects $obj
set_property -name "steps.write_bitstream.tcl.pre" -value "" -objects $obj
set_property -name "steps.write_bitstream.tcl.post" -value "" -objects $obj
set_property -name "steps.write_bitstream.args.raw_bitfile" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.mask_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.no_binary_bitfile" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.bin_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.readback_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.logic_location_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.verbose" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.more options" -value "" -objects $obj

# set the current impl run
current_run -implementation [get_runs impl_1]

puts "INFO: Project created:${project_name}"
# Create 'drc_1' gadget (if not found)
if {[string equal [get_dashboard_gadgets  [ list "drc_1" ] ] ""]} {
create_dashboard_gadget -name {drc_1} -type drc
}
set obj [get_dashboard_gadgets [ list "drc_1" ] ]
set_property -name "active_reports" -value "" -objects $obj
set_property -name "active_reports_invalid" -value "" -objects $obj
set_property -name "active_run" -value "0" -objects $obj
set_property -name "hide_unused_data" -value "1" -objects $obj
set_property -name "incl_new_reports" -value "0" -objects $obj
set_property -name "reports" -value "impl_1#impl_1_route_report_drc_0" -objects $obj
set_property -name "run.step" -value "route_design" -objects $obj
set_property -name "run.type" -value "implementation" -objects $obj
set_property -name "statistics.critical_warning" -value "1" -objects $obj
set_property -name "statistics.error" -value "1" -objects $obj
set_property -name "statistics.info" -value "1" -objects $obj
set_property -name "statistics.warning" -value "1" -objects $obj
set_property -name "view.orientation" -value "Horizontal" -objects $obj
set_property -name "view.type" -value "Graph" -objects $obj

# Create 'methodology_1' gadget (if not found)
if {[string equal [get_dashboard_gadgets  [ list "methodology_1" ] ] ""]} {
create_dashboard_gadget -name {methodology_1} -type methodology
}
set obj [get_dashboard_gadgets [ list "methodology_1" ] ]
set_property -name "active_reports" -value "" -objects $obj
set_property -name "active_reports_invalid" -value "" -objects $obj
set_property -name "active_run" -value "0" -objects $obj
set_property -name "hide_unused_data" -value "1" -objects $obj
set_property -name "incl_new_reports" -value "0" -objects $obj
set_property -name "reports" -value "impl_1#impl_1_route_report_methodology_0" -objects $obj
set_property -name "run.step" -value "route_design" -objects $obj
set_property -name "run.type" -value "implementation" -objects $obj
set_property -name "statistics.critical_warning" -value "1" -objects $obj
set_property -name "statistics.error" -value "1" -objects $obj
set_property -name "statistics.info" -value "1" -objects $obj
set_property -name "statistics.warning" -value "1" -objects $obj
set_property -name "view.orientation" -value "Horizontal" -objects $obj
set_property -name "view.type" -value "Graph" -objects $obj

# Create 'power_1' gadget (if not found)
if {[string equal [get_dashboard_gadgets  [ list "power_1" ] ] ""]} {
create_dashboard_gadget -name {power_1} -type power
}
set obj [get_dashboard_gadgets [ list "power_1" ] ]
set_property -name "active_reports" -value "" -objects $obj
set_property -name "active_reports_invalid" -value "" -objects $obj
set_property -name "active_run" -value "0" -objects $obj
set_property -name "hide_unused_data" -value "1" -objects $obj
set_property -name "incl_new_reports" -value "0" -objects $obj
set_property -name "reports" -value "impl_1#impl_1_route_report_power_0" -objects $obj
set_property -name "run.step" -value "route_design" -objects $obj
set_property -name "run.type" -value "implementation" -objects $obj
set_property -name "statistics.bram" -value "1" -objects $obj
set_property -name "statistics.clocks" -value "1" -objects $obj
set_property -name "statistics.dsp" -value "1" -objects $obj
set_property -name "statistics.gth" -value "1" -objects $obj
set_property -name "statistics.gtp" -value "1" -objects $obj
set_property -name "statistics.gtx" -value "1" -objects $obj
set_property -name "statistics.gtz" -value "1" -objects $obj
set_property -name "statistics.io" -value "1" -objects $obj
set_property -name "statistics.logic" -value "1" -objects $obj
set_property -name "statistics.mmcm" -value "1" -objects $obj
set_property -name "statistics.pcie" -value "1" -objects $obj
set_property -name "statistics.phaser" -value "1" -objects $obj
set_property -name "statistics.pll" -value "1" -objects $obj
set_property -name "statistics.pl_static" -value "1" -objects $obj
set_property -name "statistics.ps7" -value "1" -objects $obj
set_property -name "statistics.ps" -value "1" -objects $obj
set_property -name "statistics.ps_static" -value "1" -objects $obj
set_property -name "statistics.signals" -value "1" -objects $obj
set_property -name "statistics.total_power" -value "1" -objects $obj
set_property -name "statistics.transceiver" -value "1" -objects $obj
set_property -name "statistics.xadc" -value "1" -objects $obj
set_property -name "view.orientation" -value "Horizontal" -objects $obj
set_property -name "view.type" -value "Graph" -objects $obj

# Create 'timing_1' gadget (if not found)
if {[string equal [get_dashboard_gadgets  [ list "timing_1" ] ] ""]} {
create_dashboard_gadget -name {timing_1} -type timing
}
set obj [get_dashboard_gadgets [ list "timing_1" ] ]
set_property -name "active_reports" -value "" -objects $obj
set_property -name "active_reports_invalid" -value "" -objects $obj
set_property -name "active_run" -value "0" -objects $obj
set_property -name "hide_unused_data" -value "1" -objects $obj
set_property -name "incl_new_reports" -value "0" -objects $obj
set_property -name "reports" -value "impl_1#impl_1_route_report_timing_summary_0" -objects $obj
set_property -name "run.step" -value "route_design" -objects $obj
set_property -name "run.type" -value "implementation" -objects $obj
set_property -name "statistics.ths" -value "1" -objects $obj
set_property -name "statistics.tns" -value "1" -objects $obj
set_property -name "statistics.tpws" -value "1" -objects $obj
set_property -name "statistics.whs" -value "1" -objects $obj
set_property -name "statistics.wns" -value "1" -objects $obj
set_property -name "view.orientation" -value "Horizontal" -objects $obj
set_property -name "view.type" -value "Table" -objects $obj

# Create 'utilization_1' gadget (if not found)
if {[string equal [get_dashboard_gadgets  [ list "utilization_1" ] ] ""]} {
create_dashboard_gadget -name {utilization_1} -type utilization
}
set obj [get_dashboard_gadgets [ list "utilization_1" ] ]
set_property -name "active_reports" -value "" -objects $obj
set_property -name "active_reports_invalid" -value "" -objects $obj
set_property -name "active_run" -value "0" -objects $obj
set_property -name "hide_unused_data" -value "1" -objects $obj
set_property -name "incl_new_reports" -value "0" -objects $obj
set_property -name "reports" -value "synth_1#synth_1_synth_report_utilization_0" -objects $obj
set_property -name "run.step" -value "synth_design" -objects $obj
set_property -name "run.type" -value "synthesis" -objects $obj
set_property -name "statistics.bram" -value "1" -objects $obj
set_property -name "statistics.bufg" -value "1" -objects $obj
set_property -name "statistics.dsp" -value "1" -objects $obj
set_property -name "statistics.ff" -value "1" -objects $obj
set_property -name "statistics.gt" -value "1" -objects $obj
set_property -name "statistics.io" -value "1" -objects $obj
set_property -name "statistics.lut" -value "1" -objects $obj
set_property -name "statistics.lutram" -value "1" -objects $obj
set_property -name "statistics.mmcm" -value "1" -objects $obj
set_property -name "statistics.pcie" -value "1" -objects $obj
set_property -name "statistics.pll" -value "1" -objects $obj
set_property -name "statistics.uram" -value "1" -objects $obj
set_property -name "view.orientation" -value "Horizontal" -objects $obj
set_property -name "view.type" -value "Graph" -objects $obj

# Create 'utilization_2' gadget (if not found)
if {[string equal [get_dashboard_gadgets  [ list "utilization_2" ] ] ""]} {
create_dashboard_gadget -name {utilization_2} -type utilization
}
set obj [get_dashboard_gadgets [ list "utilization_2" ] ]
set_property -name "active_reports" -value "" -objects $obj
set_property -name "active_reports_invalid" -value "" -objects $obj
set_property -name "active_run" -value "0" -objects $obj
set_property -name "hide_unused_data" -value "1" -objects $obj
set_property -name "incl_new_reports" -value "0" -objects $obj
set_property -name "reports" -value "impl_1#impl_1_place_report_utilization_0" -objects $obj
set_property -name "run.step" -value "place_design" -objects $obj
set_property -name "run.type" -value "implementation" -objects $obj
set_property -name "statistics.bram" -value "1" -objects $obj
set_property -name "statistics.bufg" -value "1" -objects $obj
set_property -name "statistics.dsp" -value "1" -objects $obj
set_property -name "statistics.ff" -value "1" -objects $obj
set_property -name "statistics.gt" -value "1" -objects $obj
set_property -name "statistics.io" -value "1" -objects $obj
set_property -name "statistics.lut" -value "1" -objects $obj
set_property -name "statistics.lutram" -value "1" -objects $obj
set_property -name "statistics.mmcm" -value "1" -objects $obj
set_property -name "statistics.pcie" -value "1" -objects $obj
set_property -name "statistics.pll" -value "1" -objects $obj
set_property -name "statistics.uram" -value "1" -objects $obj
set_property -name "view.orientation" -value "Horizontal" -objects $obj
set_property -name "view.type" -value "Graph" -objects $obj

move_dashboard_gadget -name {utilization_1} -row 0 -col 0
move_dashboard_gadget -name {power_1} -row 1 -col 0
move_dashboard_gadget -name {drc_1} -row 2 -col 0
move_dashboard_gadget -name {timing_1} -row 0 -col 1
move_dashboard_gadget -name {utilization_2} -row 1 -col 1
move_dashboard_gadget -name {methodology_1} -row 2 -col 1
