vcs -full64 -sverilog -debug_all -f filelist.f -l compile.log -timescale=1ns/100ps
./simv

# rm -r ./xcelium.d
# xmverilog -f filelist.f -sv -64bit +access+rcw  -top tb_active_cta_table -stop_on_build_error -timescale 1ns/1ps 