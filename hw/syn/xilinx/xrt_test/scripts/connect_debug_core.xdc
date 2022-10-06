connect_debug_cores -master [get_cells -hierarchical -filter {NAME =~ *mgmt_debug_hub/inst/xsdbm}] -slaves [get_cells -hierarchical -filter {NAME =~ */i_ila_0}]
