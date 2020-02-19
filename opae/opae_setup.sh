

## Required tools
# gcc (>4.9)
# libjson
# python
# Quartus
# RTL Simulator (VCS or ModelSim or QuestaSim)



## Download OPAE SDK from https://github.com/OPAE/opae-sdk/archive/1.4.0-1.tar.gz
cd /nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/

## Update the following file based on /nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/libopae/plugins/ase/scripts/ase_setup_template.sh
# ./opae-sdk-1.4.0-1/libopae/plugins/ase/scripts/ase_setup_template.sh



###################################################################################################
################################### TO BE DONE EVERY TIME #########################################
###################################################################################################
## Change the shell to bash before running
bash

## Setup Environment
## Running the default script results in multiple versions of libcurl during cmake.
#source /nethome/achawda6/specialProblem/rg_intel_fpga_end_19.3.sh
source /tools/reconfig/intel/19.3/rg_intel_fpga_end_19.3.sh

## Setup the variables for using the Quartus modelsim
source /nethome/achawda6/specialProblem/modelsim_env.sh

## Run this to setup the environment variables
source /nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/libopae/plugins/ase/scripts/ase_setup_template.sh

## gcc version should be greater than 4.9 to support c++14
source /nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/libopae/plugins/ase/scripts/env_check.sh

export PATH=/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/mybuild/opaeInstall/bin:${PATH}
export FPGA_BBB_CCI_SRC=/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/intel-fpga-bbb
####################################################################################################






## Setup OPAE
mkdir mybuild
cd mybuild

## Update the directory path where you want to install OPAE
cmake .. -DBUILD_ASE=1 -DCMAKE_INSTALL_PREFIX=/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/mybuild/opaeInstall
make
make install




## Setup ASE
## Add the installed OPAE path in PATH
export PATH=/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/mybuild/opaeInstall/bin:${PATH}

## Use this version of HDL files
/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/libopae/plugins/ase/scripts/afu_sim_setup --sources=/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/libopae/plugins/ase/rtl/sources_ase_server.txt run1Build
cd run1Build/
python scripts/ipc_clean.py





## Running Sample
## Download opae-bbb from https://github.com/OPAE/intel-fpga-bbb
cd /nethome/achawda6/specialProblem/opae-sdk-1.4.0-1
git clone https://github.com/OPAE/intel-fpga-bbb
cd /nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/intel-fpga-bbb
mkdir mybuild
cd mybuild
cmake .. -DCMAKE_INSTALL_PREFIX=/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/mybuild/opaeInstall
make
make install

export FPGA_BBB_CCI_SRC=/nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/intel-fpga-bbb





## Running hello world 
cd /nethome/achawda6/specialProblem/opae-sdk-1.4.0-1/intel-fpga-bbb/samples/tutorial/01_hello_world
afu_sim_setup --source hw/rtl/sources.txt build_sim
cd build_sim
## Update libstdc++6 if it errors out
make
make sim
