#!/bin/bash

# exit when any command fails
set -e

# clear blackbox cache
rm -f blackbox.*.cache

unittest() 
{
make -C tests/unittest run
}

isa() 
{
echo "begin isa tests..."

make -C tests/riscv/isa run-simx
make -C tests/riscv/isa run-rtlsim

echo "isa tests done!"
}

regression() 
{
echo "begin regression tests..."

make -C tests/kernel run-simx
make -C tests/kernel run-rtlsim

make -C tests/regression run-simx
make -C tests/regression run-rtlsim

echo "regression tests done!"
}

opencl() 
{
echo "begin opencl tests..."

make -C tests/opencl run-simx
make -C tests/opencl run-rtlsim

echo "opencl tests done!"
}

tex()
{
echo "begin texture tests..."

CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-itoad.png -rtoad_ref_f0.png -f0 -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-itoad.png -rtoad_ref_f1.png -f1 -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-itoad.png -rtoad_ref_f2.png -f2 -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-itoad.png -rtoad_ref_f3.png -f3 -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-itoad.png -rtoad_ref_f4.png -f4 -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-itoad.png -rtoad_ref_f5.png -f5 -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-itoad.png -rtoad_ref_f6.png -f6 -g0"

CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -rsoccer_ref_g0.png -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -rsoccer_ref_g0.png -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -rsoccer_ref_g0.png -g0"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -rsoccer_ref_g1.png -g1" --perf=3
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -rsoccer_ref_g1.png -g1" --perf=3
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -rsoccer_ref_g2.png -g2"
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -rsoccer_ref_g2.png -g2"

CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -rsoccer_ref_g1.png -g1" --perf=3
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -rsoccer_ref_g1.png -g1" --perf=3
CONFIGS="-DEXT_TEX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -rsoccer_ref_g1.png -g1 -z"
CONFIGS="-DEXT_TEX_ENABLE -DTCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -rsoccer_ref_g1.png -g1"
CONFIGS="-DEXT_TEX_ENABLE -DNUM_TEX_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE" ./ci/blackbox.sh --driver=simx  --app=tex --args="-isoccer.png -rsoccer_ref_g1.png" --cores=4 --warps=1 --threads=2
CONFIGS="-DEXT_TEX_ENABLE -DNUM_TEX_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -rsoccer_ref_g1.png" --cores=1 --warps=1 --threads=2
CONFIGS="-DEXT_TEX_ENABLE -DNUM_TEX_UNITS=1 -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -rsoccer_ref_g1.png" --cores=2 --warps=1 --threads=2
CONFIGS="-DEXT_TEX_ENABLE -DNUM_TEX_UNITS=4 -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -rsoccer_ref_g1.png" --cores=2 --warps=1 --threads=2
CONFIGS="-DEXT_TEX_ENABLE -DNUM_TEX_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -rsoccer_ref_g1.png" --cores=4 --warps=1 --threads=2
CONFIGS="-DEXT_TEX_ENABLE -DNUM_TEX_UNITS=4 -DL1_DISABLE -DSM_DISABLE -DNUM_TCACHES=2" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -rsoccer_ref_g1.png" --cores=4 --warps=1 --threads=2

echo "texture tests done!"
}

rop()
{
echo "begin render output tests..."

CONFIGS="-DEXT_ROP_ENABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-rwhitebox_128.png" --perf=5
CONFIGS="-DEXT_ROP_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png" --perf=5
CONFIGS="-DEXT_ROP_ENABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png"
CONFIGS="-DEXT_ROP_ENABLE -DOCACHE_NUM_BANKS=8" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png" --perf=5
CONFIGS="-DEXT_ROP_ENABLE -DNUM_ROP_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-rwhitebox_128.png" --cores=4 --warps=1 --threads=2
CONFIGS="-DEXT_ROP_ENABLE -DNUM_ROP_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png" --cores=1 --warps=1 --threads=2
CONFIGS="-DEXT_ROP_ENABLE -DNUM_ROP_UNITS=1 -DL1_DISABLE -DSM_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png" --cores=2 --warps=1 --threads=2
CONFIGS="-DEXT_ROP_ENABLE -DNUM_ROP_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png" --cores=4 --warps=1 --threads=2
CONFIGS="-DEXT_ROP_ENABLE -DNUM_ROP_UNITS=4 -DL1_DISABLE -DSM_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png" --cores=2 --warps=1 --threads=2
CONFIGS="-DEXT_ROP_ENABLE -DNUM_ROP_UNITS=4 -DL1_DISABLE -DSM_DISABLE -DNUM_OCACHES=2" ./ci/blackbox.sh --driver=rtlsim  --app=rop --args="-rwhitebox_128.png" --cores=4 --warps=1 --threads=2

echo "render output tests done!"
}

raster()
{
echo "begin rasterizer tests..."

CONFIGS="-DEXT_RASTER_ENABLE" ./ci/blackbox.sh --driver=simx --app=raster --args="-ttriangle.cgltrace -rtriangle_ref_128.png" --perf=4
CONFIGS="-DEXT_RASTER_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=raster --args="-ttriangle.cgltrace -rtriangle_ref_128.png" --perf=4
CONFIGS="-DEXT_RASTER_ENABLE -DRCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=raster --args="-ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_RASTER_ENABLE -DRCACHE_NUM_BANKS=4" ./ci/blackbox.sh --driver=rtlsim --app=raster --args="-ttriangle.cgltrace -rtriangle_ref_128.png" --perf=4
CONFIGS="-DEXT_RASTER_ENABLE -DRASTER_TILE_LOGSIZE=4" ./ci/blackbox.sh --driver=simx --app=raster --args="-k4 -ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_RASTER_ENABLE -DRASTER_TILE_LOGSIZE=6" ./ci/blackbox.sh --driver=simx --app=raster --args="-k6 -ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_RASTER_ENABLE -DRASTER_TILE_LOGSIZE=4" ./ci/blackbox.sh --driver=rtlsim --app=raster --args="-k4 -ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_RASTER_ENABLE -DRASTER_TILE_LOGSIZE=6" ./ci/blackbox.sh --driver=rtlsim --app=raster --args="-k6 -ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DRCACHE_DISABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --cores=4 --warps=1 --threads=2
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DRCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --cores=1 --warps=1 --threads=2
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=1 -DL1_DISABLE -DSM_DISABLE -DRCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --cores=2 --warps=1 --threads=2
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=2 -DL1_DISABLE -DSM_DISABLE -DRCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --cores=4 --warps=1 --threads=2
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=2" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --warps=1 --threads=2
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=4 -DL1_DISABLE -DSM_DISABLE -DRCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --cores=2 --warps=1 --threads=2 || true
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=4 -DL1_DISABLE -DSM_DISABLE -DNUM_RCACHES=2" ./ci/blackbox.sh --driver=rtlsim  --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --cores=4 --warps=1 --threads=2 || true
CONFIGS="-DEXT_GFX_ENABLE -DRASTER_NUM_SLICES=2 -DL1_DISABLE -DSM_DISABLE -DRCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tvase.cgltrace -rbox_ref_128.png --threads=2" || true

echo "rasterizer output tests done!"
}

graphics()
{
echo "begin graphics tests..."

CONFIGS="-DEXT_IMADD_ENABLE" ./ci/blackbox.sh --driver=simx --app=imadd
CONFIGS="-DEXT_IMADD_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=imadd --args="-n32 -z"
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=2" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png"
CONFIGS="-DEXT_GFX_ENABLE -DNUM_RASTER_UNITS=2" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png"
CONFIGS="-DEXT_GFX_ENABLE -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE -DRCACHE_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --clusters=2 --cores=2 --warps=1 --threads=2
CONFIGS="-DEXT_GFX_ENABLE -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE -DRCACHE_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png" --clusters=2 --cores=2 --warps=1 --threads=2
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttriangle.cgltrace -rtriangle_ref_8.png -w8 -h8" --warps=1 --threads=2 --debug=3
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-ttriangle.cgltrace -rtriangle_ref_8.png -w8 -h8" --warps=1 --threads=2 --debug=3
CONFIGS="-DEXT_GFX_ENABLE -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE -DRCACHE_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tvase.cgltrace -rvase_ref_32.png -w32 -h32" --threads=1
CONFIGS="-DEXT_GFX_ENABLE -DIPDOM_STACK_SIZE=128" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-x -ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-y -ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-z -ttriangle.cgltrace -rtriangle_ref_128.png"
CONFIGS="-DEXT_GFX_ENABLE -DL1_DISABLE -DSM_DISABLE -DTCACHE_DISABLE -DRCACHE_DISABLE -DOCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=draw3d --args="-tvase.cgltrace -rvase_ref_32.png -w32 -h32" --threads=2 || true

echo "graphics tests done!"
}

cluster() 
{
echo "begin clustering tests..."

# warp/threads configurations
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=2 --threads=8 --app=demo
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=8 --threads=2 --app=demo
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=16 --app=demo

# cores clustering
./ci/blackbox.sh --driver=rtlsim --cores=1 --clusters=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=4 --clusters=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --app=demo --args="-n1"

# L2/L3
./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l3cache --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --l2cache --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=4 --l2cache --l3cache --app=demo --args="-n1"

echo "clustering tests done!"
}

debug()
{
echo "begin debugging tests..."

./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --perf=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --perf=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --debug=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --debug=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=opae --cores=1 --scope --app=basic --args="-t0 -n1"

echo "debugging tests done!"
}

config() 
{
echo "begin configuration tests..."

# disabling M extension
CONFIGS=-DEXT_M_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext

# disabling F extension
CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext
CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext --perf=1
CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=simx --cores=1 --app=no_mf_ext --perf=1

# disable shared memory
CONFIGS=-DSM_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem
CONFIGS=-DSM_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem --perf=1
CONFIGS=-DSM_DISABLE ./ci/blackbox.sh --driver=simx --cores=1 --app=no_smem --perf=1

# disable L1 cache
CONFIGS="-DL1_DISABLE -DSM_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DDCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm

# multiple L1 caches per cluster
CONFIGS="-DNUM_DCACHES=2 -DNUM_ICACHES=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --cores=8 --warps=1 --threads=2

# multiple FPUs per cluster
CONFIGS="-DNUM_FPU_UNITS=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --cores=8 --warps=1 --threads=2

# using Default FPU core
CONFIGS="-DDISABLE_DPI" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood

# using FPNEW FPU core
CONFIGS="-DDISABLE_DPI -DFPU_FPNEW" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood

# using AXI bus
AXI_BUS=1 ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=demo

# adjust l1 block size to match l2
CONFIGS="-DL1_LINE_SIZE=64" ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=io_addr --args="-n1"

# test cache banking
CONFIGS="-DSMEM_NUM_BANKS=4 -DDCACHE_NUM_BANKS=1" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DSMEM_NUM_BANKS=2 -DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DSMEM_NUM_BANKS=2 -DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=simx --app=sgemm
CONFIGS="-DDCACHE_NUM_BANKS=1" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=sgemm
CONFIGS="-DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=sgemm
CONFIGS="-DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=simx --cores=1 --app=sgemm

# test cache multi-porting
CONFIGS="-DDCACHE_NUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=sgemm
CONFIGS="-DDCACHE_NUM_PORTS=2" ./ci/blackbox.sh --driver=simx --cores=1 --app=sgemm
CONFIGS="-DDCACHE_NUM_PORTS=4" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=sgemm
CONFIGS="-DL2_NUM_PORTS=2 -DDCACHE_NUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=sgemm
CONFIGS="-DL2_NUM_PORTS=2 -DDCACHE_NUM_PORTS=2" ./ci/blackbox.sh --driver=simx --cores=2 --l2cache --app=sgemm

# test 128-bit MEM block
CONFIGS=-DMEM_BLOCK_SIZE=16 ./ci/blackbox.sh --driver=opae --cores=1 --app=demo

# test single-bank DRAM
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_BANKS=1" ./ci/blackbox.sh --driver=opae --cores=1 --app=demo

# test 27-bit DRAM address
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH=27" ./ci/blackbox.sh --driver=opae --cores=1 --app=demo

echo "configuration tests done!"
}

stress0() 
{
echo "begin stress0 tests..."

# test verilator reset values
CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=sgemm
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=sgemm
CONFIGS="-DDISABLE_DPI -DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=dogfood
CONFIGS="-DDISABLE_DPI -DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=dogfood
CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr
CONFIGS="-DVERILATOR_RESET_VALUE=1 -DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=draw3d --args="-tbox.cgltrace -rbox_ref_128.png"
CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=opae --app=printf
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=opae --app=printf

echo "stress0 tests done!"
}

stress1() 
{
echo "begin stress1 tests..."

./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --clusters=2 --l3cache --app=sgemm --args="-n256"

echo "stress1 tests done!"
}

show_usage()
{
    echo "Vortex Regression Test" 
    echo "Usage: $0 [-unittest] [-isa] [-regression] [-opencl] [-tex] [-rop] [-raster] [-graphics] [-cluster] [-debug] [-config] [-stress[#n]] [-all] [-h|--help]"
}

start=$SECONDS

while [ "$1" != "" ]; do
    case $1 in
        -unittest ) unittest
                ;;
        -isa ) isa
                ;;
        -regression ) regression
                ;;
        -opencl ) opencl
                ;;
        -tex ) tex
                ;;
        -rop ) rop
                ;;
        -raster ) raster
                ;;
        -graphics ) graphics
                ;;
        -cluster ) cluster
                ;;
        -debug ) debug
                ;;
        -config ) config
                ;;
        -stress0 ) stress0
                ;;
        -stress1 ) stress1
                ;;
        -stress ) stress0
                  stress1
                ;;
        -all ) unittest               
               isa
               regression
               opencl
               graphics
               tex
               rop
               raster               
               cluster
               debug
               config
               stress0
               stress1
                ;;
        -h | --help ) show_usage
                      exit
                ;;
        * )           show_usage
                      exit 1
    esac
    shift
done

echo "Regression completed!"

duration=$(( SECONDS - start ))
awk -v t=$duration 'BEGIN{t=int(t*1000); printf "Elapsed Time: %d:%02d:%02d\n", t/3600000, t/60000%60, t/1000%60}'
