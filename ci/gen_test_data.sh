#!/bin/bash

# exit when any command fails
set -e

graphics()
{
echo "begin graphics data generation..."

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -osoccer_ref_g0.png -g0"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -osoccer_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -osoccer_ref_g2.png -g2"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette4.png -opalette4_ref_g0.png -g0"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette4.png -opalette4_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette4.png -opalette4_ref_g2.png -g2"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette16.png -opalette16_ref_g0.png -g0"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette16.png -opalette16_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette16.png -opalette16_ref_g2.png -g2"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette64.png -opalette64_ref_g0.png -g0"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette64.png -opalette64_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-ipalette64.png -opalette64_ref_g2.png -g2"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-owhitebox_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-owhitebox_64.png -w64 -h64"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-owhitebox_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-owhitebox_16.png -w16 -h16"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-owhitebox_8.png -w8 -h8"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=raster --args="-ttriangle.cgltrace -otriangle_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=raster --args="-ttriangle.cgltrace -otriangle_ref_64.png -w64 -h64"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=raster --args="-ttriangle.cgltrace -otriangle_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=raster --args="-ttriangle.cgltrace -otriangle_ref_16.png -w16 -h16"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=raster --args="-ttriangle.cgltrace -otriangle_ref_8.png -w8 -h8"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttriangle.cgltrace -otriangle_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttriangle.cgltrace -otriangle_ref_64.png -w64 -h64"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttriangle.cgltrace -otriangle_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttriangle.cgltrace -otriangle_ref_16.png -w16 -h16"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttriangle.cgltrace -otriangle_ref_8.png -w8 -h8"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tcoverflow.cgltrace -ocoverflow_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tcoverflow.cgltrace -ocoverflow_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tevilskull.cgltrace -oevilskull_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tevilskull.cgltrace -oevilskull_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tfilmtv.cgltrace -ofilmtv_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tfilmtv.cgltrace -ofilmtv_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tmouse.cgltrace -omouse_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tmouse.cgltrace -omouse_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tpolybump.cgltrace -opolybump_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tpolybump.cgltrace -opolybump_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tskybox.cgltrace -oskybox_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tskybox.cgltrace -oskybox_ref_32.png -w32 -h32"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tvase.cgltrace -ovase_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tvase.cgltrace -ovase_ref_32.png -w32 -h32"

CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttekkaman.cgltrace -otekkaman_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tbox.cgltrace -obox_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tcarnival.cgltrace -ocarnival_ref_128.png -w128 -h128"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tscene.cgltrace -oscene_ref_128.png -w128 -h128"

echo "end graphics data generation..."
}

show_usage()
{
    echo "Generate Test Data"
    echo "Usage: $0 [[-graphics] [-h|--help]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -graphics ) graphics
                ;;
        -h | --help ) show_usage
                      exit
                      ;;
        * )           show_usage
                      exit 1
    esac
    shift
done
