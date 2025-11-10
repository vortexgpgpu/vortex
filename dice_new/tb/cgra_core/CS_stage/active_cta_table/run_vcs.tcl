vcs -full64 -sverilog -f filelist.f \
    -debug_access+pp+all -kdb -lca +vpi \
    +define+FSDB \
    -o simv
./simv
