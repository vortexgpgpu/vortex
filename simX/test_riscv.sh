echo start > results.txt

# echo ../kernel/vortex_test.hex
make clean
make
cd obj_dir && ./Vcache_simX -E -a rv32i --core ../../runtime/mains/dev/vx_dev_main.hex  -s -b 1> emulator.debug
