#!/bin/bash

while getopts c: flag
do
  case "${flag}" in
    c) i=${OPTARG};; #cores: 1, 2, 4, 8, 16
  esac
done

if [[ ! "$i" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'Invalid parameter for argument -c (1, 2, 4, 8, or 16 expected)'
  exit 1
fi

cd "../../hw/syn/opae/build_fpga_${i}c"

printf "y\ny\ny\n" | PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs > /dev/null

fpgasupdate vortex_afu_unsigned_ssl.gbs
