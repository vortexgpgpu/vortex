#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

simple()
{
echo "begin rop tests"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_8.png -w8 -h8" --perf | grep 'PERF' > ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_16.png -w16 -h16" --perf | grep 'PERF' >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_32.png -w32 -h32" --perf  | grep 'PERF' >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_64.png -w64 -h64" --perf | grep 'PERF' >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png -w128 -h128" --perf | grep 'PERF' >> ./perf/rop/rop_perf.log

echo "rop tests done!"
}

depth_stencil()
{
echo "begin rop tests (with depth-stencil)"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_8.png -w8 -h8 -d" --perf > ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_16.png -w16 -h16 -d" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_32.png -w32 -h32 -d" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_64.png -w64 -h64 -d" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png -w128 -h128 -d" --perf >> ./perf/rop/rop_perf.log

echo "rop tests done!"
}

blend()
{
echo "begin rop tests (with blend)"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_8.png -w8 -h8 -b" --perf > ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_16.png -w16 -h16 -b" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_32.png -w32 -h32 -b" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_64.png -w64 -h64 -b" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png -w128 -h128 -b" --perf >> ./perf/rop/rop_perf.log

echo "rop tests done!"
}

depth_stencil_blend()
{
echo "begin rop tests (with depth-stencil & blend)"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_8.png -w8 -h8 -b -d" --perf > ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_16.png -w16 -h16 -b -d" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_32.png -w32 -h32 -b -d" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_64.png -w64 -h64 -b -d" --perf >> ./perf/rop/rop_perf.log
echo -e "\n**************************************\n" >> ./perf/rop/rop_perf.log
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=rop --args="-rwhitebox_128.png -w128 -h128 -b -d" --perf >> ./perf/rop/rop_perf.log

echo "rop tests done!"
}

usage()
{
    echo "usage: [-d] [-b] [-db] [-h|--help]"
}

case $1 in
    -d ) depth_stencil
            ;;
    -b ) blend
            ;;
    -db ) depth_stencil_blend
            ;;
    -h | --help ) usage
                    ;;
    * ) simple
        ;;             
esac
shift