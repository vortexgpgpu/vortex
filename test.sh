cd build
../configure --xlen=64 --tooldir=$HOME/tools
source ./ci/toolchain_env.sh
CONFIGS="-DVM_ENABLE" ./ci/blackbox.sh --driver=simx --app=demo --rebuild=1 --debug=1
