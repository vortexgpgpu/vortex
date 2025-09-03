#!/bin/bash 

if [ ! -d "log" ]; then
  mkdir log
fi

folders=("backprop"
"bfs"
"blackscholes"
"b+tree"
"cfd"
"conv3"
"dotproduct"
"gaussian"
"hotspot3D"
"kmeans"
"lavaMD"
"nearn"
"lbm"
"pathfinder"
"psum"
"saxpy"
"sfilter"
"sgemm"
"sgemm2"
"sgemm3"
"spmv"
"srad"
"transpose"
"vecadd"
"psort")


for folder in "${folders[@]}"; do
  echo "Processing app: $folder"
  ./subtest.sh $folder &
done 

wait
