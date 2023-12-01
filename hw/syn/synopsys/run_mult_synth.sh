#!/bin/bash

# Copyright © 2019-2023
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set top_level = Vortex

source /tools/synopsys/synthesis/j201409/cshrc.syn
set cur_dir = `pwd`
echo $cur_dir

for number_of_warps in 2 4 8 16 32; do
    for number_of_threads in 2 4 8 16 32; do
        
            echo "Warp Count: $number_of_warps  Thread Count: $number_of_threads Launched"
            echo "\`define NT $number_of_threads" > ../rtl/VX_define_synth.v
            echo "\`define NW $number_of_warps" >> ../rtl/VX_define_synth.v
            make dc | tee run.log 1>/dev/null
            sleep 30
            moved_filename="${number_of_warps}_Warps__${number_of_threads}_threads__400MHz.log"
            mv ./vortex_syn.log ../../$moved_filename
            sleep 30




            echo "Warp Count: $number_of_warps  Thread Count: $number_of_threads Finished"
    done
done


echo "Done!"
