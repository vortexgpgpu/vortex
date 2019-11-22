#!/bin/bash
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
