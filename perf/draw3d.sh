#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

# clear blackbox cache
rm -f blackbox.*.cache

# draw3d benchmarks
echo "-------------------------"
echo "draw3d vase benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tvase.cgltrace -w8 -h8"
echo "-------------------------"
echo "draw3d filmtv benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tfilmtv.cgltrace -w8 -h8"
echo "-------------------------"
echo "draw3d skybox benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tskybox.cgltrace -w8 -h8"
echo "-------------------------"
echo "draw3d coverflow benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tcoverflow.cgltrace -w8 -h8"
echo "-------------------------"
echo "draw3d evilskull benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tevilskull.cgltrace -w8 -h8"
echo "-------------------------"
echo "draw3d polybump benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tpolybump.cgltrace -w8 -h8"
echo "-------------------------"
echo "draw3d tekkaman benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttekkaman.cgltrace -w8 -h8"
echo "-------------------------"
echo "draw3d carnival benchmark"
echo "-------------------------"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-tcarnival.cgltrace -w8 -h8"